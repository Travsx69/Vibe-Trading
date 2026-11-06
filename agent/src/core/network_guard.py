"""Deny network access while generated strategy code is running.

The backtest child needs network access for its *own* data fetch: ``backtest.
runner.main`` downloads the price panel before it ever instantiates the strategy
(``fetch_data_map`` -> ``engine_cls()`` -> ``generate``).  Cutting the whole
child off would break every run that is not served from a warm cache, so the
portable layer here arms only around the two points where code we did not write
gets to execute: constructing ``SignalEngine`` and calling its ``generate``.

The mechanism is a CPython audit hook, not a patched ``socket`` module:

* an audit hook is consulted inside the C implementations, so
  ``import _socket`` (the C module behind ``socket``) is covered too;
* the hook cannot be removed once installed -- ``sys.addaudithook`` has no
  counterpart -- so a strategy cannot un-arm the process;
* it is process-wide, so a thread the strategy starts is covered as well.

Raising :class:`NetworkAccessDenied` aborts the call before the kernel sees it.
It deliberately subclasses ``RuntimeError`` rather than ``OSError`` so retry
helpers cannot mistake it for a transient socket error and try again.

Strictness is selected by ``VIBE_TRADING_SANDBOX_NETWORK``:

* unset / ``auto`` -- arm the execution-phase guard, child keeps its data fetch;
* ``required`` -- the runner additionally puts the *whole* child in a fresh
  network namespace (see ``src.core.runner``), so nothing in the child can
  reach the network at all.  A live data fetch will therefore fail; this mode
  is for deployments that pre-warm the cache or use a no-network source;
* ``off`` -- install nothing.  Only for a deployment that accepts that
  generated strategy code can reach the network.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator

ENV_VAR = "VIBE_TRADING_SANDBOX_NETWORK"

#: Audit events that fire when a process is about to move bytes or resolve a
#: name.  ``socket.__new__`` is deliberately absent: allocating the object is
#: not the boundary.  Every path that can reach a peer does go through one of
#: these -- including an ``AF_UNIX`` connect, which can reach a host daemon, so
#: those are denied as well.  Generated strategy code has no legitimate socket
#: peer.
#:
#: One platform note on ``socket.socketpair()``: on POSIX it is a real
#: ``socketpair(2)`` syscall and stays usable.  CPython emulates it on Windows
#: by binding and connecting a loopback listener, so there it trips
#: ``socket.bind``/``socket.connect`` and is denied.  That is the safe side of
#: the asymmetry -- the denial is inside the armed block only, and no strategy
#: template uses socketpair.
_DENY_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.connect_ex",
        "socket.bind",
        "socket.sendto",
        "socket.sendmsg",
        "socket.getaddrinfo",
        "socket.getnameinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getservbyname",
        "socket.getservbyport",
        "urllib.Request",
    }
)

# ``armed`` is a counter, not a bool, so nested ``deny_network()`` blocks leave
# the guard armed until the outermost one exits.
_STATE = {"installed": False, "armed": 0}


class NetworkAccessDenied(RuntimeError):
    """Raised when generated strategy code tries to touch the network."""


def network_mode() -> str:
    """Return the configured mode: ``"required"``, ``"off"`` or ``"auto"``."""
    raw = os.environ.get(ENV_VAR, "").strip().lower()  # noqa: env-gate — sandbox hardening mode, not app config
    if raw in {"off", "0", "false", "no"}:
        return "off"
    if raw in {"required", "require", "1", "true", "yes"}:
        return "required"
    return "auto"


def _audit_hook(event: str, _args: tuple[object, ...]) -> None:
    if not _STATE["armed"]:
        return
    if event in _DENY_EVENTS:
        raise NetworkAccessDenied(
            f"generated strategy code may not reach the network ({event}); "
            "all data must come from the backtest data snapshot"
        )


def install() -> bool:
    """Install the audit hook.  Idempotent; a no-op in ``off`` mode.

    Installing does *not* arm the guard.  Arming is scoped to the strategy call
    by :func:`deny_network`, which keeps the child's own data fetch working.

    Returns:
        True when a hook is installed (now or previously), False in ``off``
        mode, where generated code is left unguarded on purpose.
    """
    if network_mode() == "off":
        return False
    if not _STATE["installed"]:
        sys.addaudithook(_audit_hook)
        _STATE["installed"] = True
    return True


def is_installed() -> bool:
    """Return whether the audit hook has been installed in this process."""
    return bool(_STATE["installed"])


def is_armed() -> bool:
    """Return whether network calls are currently being denied."""
    return _STATE["armed"] > 0


@contextmanager
def deny_network() -> Iterator[None]:
    """Deny network access for the duration of the block.

    Installs the hook first so a manual ``python -m backtest.runner`` run is
    guarded even though it never went through the runner bootstrap.  In ``off``
    mode this is a transparent pass-through.
    """
    if not install():
        yield
        return
    _STATE["armed"] += 1
    try:
        yield
    finally:
        _STATE["armed"] -= 1
