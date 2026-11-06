"""Private strategy IPC using fresh pipes, never the MCP server's stdin.

The existing strategy sandbox still validates and executes the source. The two
directions use different encodings on purpose:

* parent -> child is pickle: the payload, account state and DataFrames come
  from the trusted parent, and the child is the one that unpickles them.
* child -> parent is one JSON array per line: the child runs LLM-generated
  strategy code, so nothing it writes may be unpickled (a crafted pickle would
  execute in the parent, which holds the full environment). Its messages are
  only ``["ready"]``, ``["day", day, orders]``, ``["error", day, name]`` and
  ``["done"]`` built from strings, finite floats and null.

The child also gets an allowlisted environment and an empty private working
directory, so strategy code can read neither the parent's secrets nor
relative files such as ``agent/.env``. File access while strategy code runs
is confined by ``src.core.fs_guard``; with ``VIBE_TRADING_SANDBOX_FS=required``
the child additionally starts in a Linux mount namespace (``_fs_isolation_prefix``).
"""

from __future__ import annotations

import json
import os
import pickle
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

# Largest single child message; orders for one day are far below this.
_MAX_CHILD_MESSAGE_BYTES = 16 * 1024 * 1024


def decode_child_message(line: bytes) -> tuple[Any, ...]:
    """Parse one child->parent line, refusing anything but a tagged JSON array.

    Args:
        line: One newline-terminated line read from the child's stdout.

    Returns:
        The message as a tuple whose first element is its string tag.

    Raises:
        ValueError: If the line is truncated, oversized, or not a tagged array.
    """
    if not line.endswith(b"\n"):
        raise ValueError("strategy message truncated or larger than the limit")
    message = json.loads(line)
    if not (isinstance(message, list) and message and isinstance(message[0], str)):
        raise ValueError("strategy message must be a JSON array with a string tag")
    return tuple(message)


def encode_child_message(message: Any) -> bytes:
    """Serialize one child->parent message as a JSON line."""
    return json.dumps(list(message), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


class StrategyConnection:
    """Connection-compatible messages with a bounded parent-side poll.

    ``parent=True`` reads JSON lines from the child and writes pickle to it;
    the child side does the reverse.
    """

    def __init__(self, reader, writer, *, parent: bool = False, workdir: str | None = None):
        self.reader = reader
        self.writer = writer
        self.parent = parent
        self.workdir = workdir
        self.messages: queue.Queue = queue.Queue()
        self.pending = None
        self.thread = None
        if parent:
            self.thread = threading.Thread(target=self._read, name="paper-strategy-ipc", daemon=True)
            self.thread.start()

    def _read(self) -> None:
        try:
            while True:
                line = self.reader.readline(_MAX_CHILD_MESSAGE_BYTES + 1)
                if not line:
                    raise EOFError
                self.messages.put((True, decode_child_message(line)))
        except Exception:
            self.messages.put((False, None))

    def send(self, message: Any) -> None:
        if self.parent:
            pickle.dump(message, self.writer, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.writer.write(encode_child_message(message))
        self.writer.flush()

    def poll(self, timeout: float) -> bool:
        if self.pending is None:
            try:
                self.pending = self.messages.get(timeout=timeout)
            except queue.Empty:
                return False
        return True

    def recv(self) -> Any:
        if self.thread is None:
            return pickle.load(self.reader)  # child side: data from the trusted parent
        success, message = self.pending if self.pending is not None else self.messages.get()
        self.pending = None
        if not success:
            raise EOFError("strategy process closed its protocol")
        return message

    def close(self) -> None:
        # Caller reaps the child first, releasing any blocked pipe read.
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        for stream in (self.writer, self.reader):
            try:
                stream.close()
            except OSError:
                pass
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)


class SandboxFilesystemIsolationError(RuntimeError):
    """``VIBE_TRADING_SANDBOX_FS=required`` was asked for but cannot be provided."""


def _fs_isolation_script(workdir: str) -> str:
    """Build the sh script that hides the host's private files, then execs the child.

    Run under ``unshare --user --map-root-user --mount``. ``$HOME`` and the
    runtime root are each replaced by a fresh tmpfs; the interpreter, venv,
    ``sys.path`` and workdir directories that live beneath them are bound back
    in (read-only, except the workdir), and every ``.env*`` file directly in a
    ``sys.path`` entry is covered with ``/dev/null``.
    """
    from src.config.paths import get_runtime_root

    def real(path: Any) -> str:
        return os.path.realpath(str(path))

    def inside(path: str, root: str) -> bool:
        return path == root or path.startswith(root.rstrip("/") + "/")

    hidden: list[str] = []
    for root in sorted({real(Path.home()), real(get_runtime_root())}, key=len):
        if not any(inside(root, kept) for kept in hidden):
            hidden.append(root)
    code_dirs = [real(entry) for entry in sys.path if entry and os.path.isdir(entry)]
    keep = {real(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)} | set(code_dirs)
    workdir = real(workdir)
    q = shlex.quote
    lines = ["set -e"]
    for index, root in enumerate(hidden):
        stage = f'"$stage{index}"'
        lines += [f"stage{index}=$(mktemp -d)", f"mount -t tmpfs -o mode=0755 tmpfs {stage}"]
        for path in sorted(p for p in keep | {workdir} if inside(p, root) and p != root):
            target = f"{stage}/{q(os.path.relpath(path, root))}"
            lines += [f"mkdir -p {target}", f"mount --rbind {q(path)} {target}"]
            if path != workdir:
                lines.append(f"mount -o remount,bind,ro {target}")
        lines.append(f"mount --move {stage} {q(root)}")
    for directory in code_dirs:
        for name in sorted(os.listdir(directory)):
            if name.startswith(".env") and os.path.isfile(os.path.join(directory, name)):
                lines.append(f"mount --bind /dev/null {q(os.path.join(directory, name))}")
    lines.append('exec "$@"')
    return "\n".join(lines)


def _fs_isolation_prefix(workdir: str) -> list[str]:
    """Return the argv prefix for ``required`` mode, or raise if the host cannot do it.

    Probed with the real script, so a host that allows user namespaces but not
    the mounts still refuses instead of starting an unconfined child.
    """
    unshare, sh = shutil.which("unshare"), shutil.which("sh")
    if not sys.platform.startswith("linux") or unshare is None or sh is None:
        raise SandboxFilesystemIsolationError(
            "VIBE_TRADING_SANDBOX_FS=required needs Linux with 'unshare'; refusing to run "
            "strategy code without a filesystem boundary. Set it to 'auto' to rely on the "
            "in-process guard only."
        )
    prefix = [unshare, "--user", "--map-root-user", "--mount", "--", sh, "-c",
              _fs_isolation_script(workdir), "vt-fs-sandbox"]
    try:
        probe = subprocess.run([*prefix, sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxFilesystemIsolationError(f"filesystem sandbox probe failed: {exc}") from exc
    if probe.returncode != 0:
        raise SandboxFilesystemIsolationError(
            "VIBE_TRADING_SANDBOX_FS=required, but the mount-namespace sandbox could not be "
            "set up (user namespaces disabled or mounts refused): "
            + probe.stderr.decode(errors="replace").strip()[-300:]
        )
    return prefix


def start_strategy_process(payload: dict[str, Any]):
    """Start an isolated interpreter and deliver the validated source snapshot.

    Args:
        payload: Source, filename, days and universe already checked by caller.

    Returns:
        The child process and its private bidirectional message connection.
    """
    from src.core.fs_guard import fs_mode
    from src.core.runner import _copy_runtime_env

    # Preserve import paths for editable installs and standalone MCP hosts.
    # -B keeps research execution from writing bytecode into a caller's cwd.
    bootstrap = (
        f"import sys; sys.path[:] = {sys.path!r}; "
        "from src.tools.paper_trade.strategy_process import worker_main; worker_main()"
    )
    workdir = tempfile.mkdtemp(prefix="vt-paper-strategy-")
    try:
        prefix = _fs_isolation_prefix(workdir) if fs_mode() == "required" else []
        process = subprocess.Popen(
            [*prefix, sys.executable, "-B", "-u", "-c", bootstrap],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            close_fds=True, cwd=workdir, env=_copy_runtime_env(),
        )
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    connection = StrategyConnection(process.stdout, process.stdin, parent=True, workdir=workdir)
    try:
        connection.send(payload)
    except BaseException:
        process.kill()
        process.wait()
        connection.close()
        raise
    return process, connection


def worker_main() -> None:
    """Keep strategy/import prints out of the protocol stream."""
    connection = StrategyConnection(sys.stdin.buffer, sys.stdout.buffer)
    sys.stdout = sys.stderr
    payload = connection.recv()
    from .strategy_exec import _child_main

    _child_main(connection, payload)
