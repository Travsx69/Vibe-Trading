"""Confine file access while generated strategy code is running.

A private working directory decides where *relative* paths land; it does not
stop ``open('/home/me/.vibe-trading/okx.json')``. This module is the portable
layer of the paper-strategy child's filesystem boundary, built the same way as
``src.core.network_guard``: a CPython audit hook, armed only while strategy code
executes (module load and each ``generate_signals`` call), that cannot be
removed once installed.

While armed, a path event is allowed only when its resolved path is:

* under the child's own working directory (read and write);
* under the interpreter / venv / timezone-data roots (read only) -- library
  code and data that imports may load lazily;
* a code file (``.py``, ``.pyc``, extension module) under a ``sys.path``
  entry (read only). Non-code files there -- ``agent/.env`` above all -- stay
  out of reach.

Audit hooks are not a complete sandbox (PEP 578), so on Linux
``VIBE_TRADING_SANDBOX_FS=required`` adds the kernel layer in
``src.tools.paper_trade.strategy_process``: a mount namespace in which the real
``$HOME`` and runtime root are replaced by an empty tmpfs.

Modes (``VIBE_TRADING_SANDBOX_FS``): unset / ``auto`` -- this hook only;
``required`` -- hook plus mount namespace, refusing to start without it;
``off`` -- neither.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

ENV_VAR = "VIBE_TRADING_SANDBOX_FS"

_CODE_SUFFIXES = (".py", ".pyc", ".pyd", ".so", ".dll", ".pth", ".typed")
# Events whose first (and, for rename/link, second) argument is a path.
_PATH_EVENTS = frozenset(
    {
        "open", "os.listdir", "os.scandir", "os.remove", "os.rename", "os.rmdir",
        "os.mkdir", "os.symlink", "os.link", "os.truncate", "os.chmod", "os.chown",
        "os.utime", "shutil.copyfile", "shutil.rmtree", "shutil.move",
    }
)
_TWO_PATH_EVENTS = frozenset({"os.rename", "os.symlink", "os.link", "shutil.copyfile", "shutil.move"})

_STATE: dict = {"installed": False, "armed": 0, "busy": False, "rw": (), "ro": (), "code": ()}


class FileAccessDenied(RuntimeError):
    """Raised when generated strategy code reaches outside its sandbox.

    A ``RuntimeError``, not ``OSError``/``PermissionError``, so ``except OSError``
    fallbacks in strategy code cannot swallow it and carry on.
    """


def fs_mode() -> str:
    """Return the configured mode: ``"required"``, ``"off"`` or ``"auto"``."""
    raw = os.environ.get(ENV_VAR, "").strip().lower()  # noqa: env-gate — sandbox hardening mode, not app config
    if raw in {"off", "0", "false", "no"}:
        return "off"
    if raw in {"required", "require", "1", "true", "yes"}:
        return "required"
    return "auto"


def _real(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _is_write(mode: object, flags: object) -> bool:
    if isinstance(mode, str):
        return any(ch in mode for ch in "wax+")
    if isinstance(flags, int):  # os.open passes mode=None and O_* flags
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    return True  # unknown: treat as a write


def _allowed(raw: object, event: str, writes: bool) -> bool:
    if isinstance(raw, int):  # an already-open fd, not a path
        return True
    try:
        path = _real(os.fsdecode(os.fspath(raw)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if _under(path, _STATE["rw"]):
        return True
    if writes or event not in {"open", "os.listdir", "os.scandir"}:  # mutations are rw-only
        return False
    if _under(path, _STATE["ro"]):
        return True
    if not _under(path, _STATE["code"]):
        return False
    # The import system lists sys.path directories; reading is code files only.
    return event != "open" or path.endswith(_CODE_SUFFIXES)


def _audit_hook(event: str, args: tuple) -> None:
    if not _STATE["armed"] or _STATE["busy"] or event not in _PATH_EVENTS or not args:
        return
    _STATE["busy"] = True  # realpath must not re-enter the hook
    try:
        paths = args[:2] if event in _TWO_PATH_EVENTS else args[:1]
        if event in {"os.listdir", "os.scandir"} and args[0] is None:
            paths = (".",)
        # "open" args are (path, mode, flags); os.open passes mode=None.
        writes = event == "open" and _is_write(
            args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None
        )
        for raw in paths:
            if raw is not None and not _allowed(raw, event, writes):
                raise FileAccessDenied(
                    f"generated strategy code may not touch files outside its sandbox ({event})"
                )
    finally:
        _STATE["busy"] = False


def _read_only_roots() -> list[str]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix}
    try:
        import zoneinfo

        roots.update(zoneinfo.TZPATH)
    except ImportError:  # pragma: no cover - stdlib on 3.9+
        pass
    return [_real(root) for root in roots if root]


def install() -> bool:
    """Install the hook (idempotent); False in ``off`` mode."""
    if fs_mode() == "off":
        return False
    if not _STATE["installed"]:
        sys.addaudithook(_audit_hook)
        _STATE["installed"] = True
    return True


@contextmanager
def confine_files(workdir: str | os.PathLike[str]) -> Iterator[None]:
    """Deny file access outside ``workdir`` and read-only library roots for the block."""
    if not install():
        yield
        return
    if not _STATE["armed"]:
        _STATE["rw"] = (_real(os.fspath(workdir)),)
        _STATE["ro"] = tuple(_read_only_roots())
        _STATE["code"] = tuple(_real(entry) for entry in sys.path if entry and Path(entry).is_dir())
    _STATE["armed"] += 1
    try:
        yield
    finally:
        _STATE["armed"] -= 1
