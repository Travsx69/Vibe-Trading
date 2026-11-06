"""Regression tests for the P0 day-0 security fixes.

Covers: no ``.env`` reload in the backtest child (F3), pathlib reads confined
to the run dir, broker host pinning and the agent/MCP local-gateway rule (N1),
paper profiles refusing a live config (F7), child->parent paper IPC that never
unpickles (M4) with a scrubbed child environment, and the P1.1 network guard
that denies sockets while generated strategy code runs.

Every credential here is a dummy canary; nothing touches a network or broker.
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import os
import pickle
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from backtest import runner as bt_runner
from backtest.runner import _load_module_from_file
from src.core import network_guard
from src.core import runner as sandbox_runner
from src.trading.endpoint_policy import is_broker_url, is_local_gateway_host

CANARY = "CANARY-" + uuid.uuid4().hex


def _write_engine(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "signal_engine.py"
    path.write_text(
        "\n".join(
            [
                '"""Generated signal engine."""',
                "from pathlib import Path",
                "class SignalEngine:",
                "    def generate(self, *args, **kwargs):",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# F3: the backtest child must never re-read agent/.env
# --------------------------------------------------------------------------- #


def test_runner_main_never_calls_load_dotenv() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(bt_runner.main)))
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "load_dotenv" not in calls


def test_backtest_tool_runs_child_in_run_dir_not_agent_root() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "tools" / "backtest_tool.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    execute_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "runner"
    ]
    assert len(execute_calls) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in execute_calls[0].keywords}
    assert keywords["cwd"] == "run_path"
    assert keywords["pythonpath"] == "agent_root"


def test_runner_child_cannot_see_dotenv_by_relative_path(tmp_path: Path) -> None:
    pytest.importorskip("rich")
    from src.core.runner import Runner

    # A fake agent root holding a canary .env and an importable module, and a
    # run dir elsewhere: the child must import from pythonpath but run in run_dir.
    agent_root = tmp_path / "agent"
    (agent_root / "probe_pkg").mkdir(parents=True)
    (agent_root / "probe_pkg" / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    (agent_root / ".env").write_text(f"OPENAI_API_KEY={CANARY}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    entry = tmp_path / "probe.py"
    entry.write_text(
        "import json, os, probe_pkg\n"
        "print(json.dumps({'cwd': os.getcwd(), 'dotenv_visible': os.path.exists('.env'),"
        " 'value': probe_pkg.VALUE, 'canary_in_env': any("
        f"{CANARY!r} in v for v in os.environ.values())}}))\n",
        encoding="utf-8",
    )

    result = Runner(timeout=60).execute(entry, run_dir, cwd=run_dir, pythonpath=agent_root)

    assert result.success, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(report["cwd"]).resolve() == run_dir.resolve()
    assert report == {**report, "dotenv_visible": False, "value": 7, "canary_in_env": False}


def test_runner_child_keeps_data_source_config_from_env_and_dotenv(tmp_path, monkeypatch) -> None:
    pytest.importorskip("rich")
    pytest.importorskip("dotenv")
    from src.core.runner import Runner
    from src.providers import llm

    # GILDATA_TOKEN exported; TickerAll only in .env next to a secret that must stay out.
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "TICKERALL_API_KEY=dummy-ta-key\nTICKERALL_ACCOUNT_ID=dummy-ta-acct\n"
        f"OPENAI_API_KEY={CANARY}\nGILDATA_TOKEN=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm, "_ENV_CANDIDATES", [dotenv])
    monkeypatch.setenv("GILDATA_TOKEN", "dummy-gildata")
    monkeypatch.delenv("TICKERALL_API_KEY", raising=False)
    monkeypatch.delenv("TICKERALL_ACCOUNT_ID", raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    entry = tmp_path / "probe.py"
    entry.write_text(
        "import json, os\n"
        "keys = ('GILDATA_TOKEN', 'TICKERALL_API_KEY', 'TICKERALL_ACCOUNT_ID', 'OPENAI_API_KEY')\n"
        "print(json.dumps({k: os.environ.get(k) for k in keys}))\n",
        encoding="utf-8",
    )

    result = Runner(timeout=60).execute(entry, run_dir, cwd=run_dir)

    assert result.success, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "GILDATA_TOKEN": "dummy-gildata",
        "TICKERALL_API_KEY": "dummy-ta-key",
        "TICKERALL_ACCOUNT_ID": "dummy-ta-acct",
        "OPENAI_API_KEY": None,
    }


def test_runtime_env_allowlist_keeps_source_order_drops_secrets() -> None:
    pytest.importorskip("rich")
    from src.core.runner import _is_runtime_env_key_allowed

    assert _is_runtime_env_key_allowed("MARKET_DATA_ORDER_A_SHARE")
    for secret in ("OPENAI_API_KEY", "API_AUTH_KEY", "TAP_AGENT_KEY", "ETORO_API_KEY", "LONGBRIDGE_APP_SECRET"):
        assert not _is_runtime_env_key_allowed(secret)


# --------------------------------------------------------------------------- #
# Scanner: pathlib reads confined to Path("<relative literal>")
# --------------------------------------------------------------------------- #

_FORBIDDEN_READS = [
    ("dunder_file_dotenv", "        Path(__file__).parents[2].joinpath('.env').read_text()\n        return []"),
    ("absolute_read_text", "        Path('/etc/passwd').read_text()\n        return []"),
    ("home_read_bytes", "        Path('~/.vibe-trading/okx.json').read_bytes()\n        return []"),
    ("parent_open", "        Path('../.env').open()\n        return []"),
    ("variable_receiver", "        p = Path('x')\n        p.read_text()\n        return []"),
    ("windows_drive", "        Path('C:/Users/x/.env').read_text()\n        return []"),
]


@pytest.mark.parametrize("case_id,body", _FORBIDDEN_READS, ids=[c[0] for c in _FORBIDDEN_READS])
def test_scanner_rejects_pathlib_reads_outside_run_dir(tmp_path, case_id, body) -> None:
    with pytest.raises(ValueError, match="not allowed inside generated strategy code"):
        _load_module_from_file(_write_engine(tmp_path, body), f"se_{uuid.uuid4().hex}")


@pytest.mark.parametrize(
    "body",
    [
        "        Path('data.csv').read_text()\n        return []",
        "        Path('inputs/prices.bin').read_bytes()\n        return []",
        "        Path('data.csv').open()\n        return []",
    ],
)
def test_scanner_allows_relative_literal_reads(tmp_path, body) -> None:
    _load_module_from_file(_write_engine(tmp_path, body), f"se_{uuid.uuid4().hex}")


# --------------------------------------------------------------------------- #
# N1: broker hosts pinned; agent/MCP host override only for local gateways
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url,ok",
    [
        ("https://www.okx.com", True),
        ("https://my.okx.com:443", True),
        ("https://okx.com", True),
        ("http://www.okx.com", False),
        ("https://attacker.example", False),
        ("https://www.okx.com.attacker.example", False),
        ("https://evilokx.com", False),
        ("https://user:pw@www.okx.com", False),
        ("https://www.okx.com@attacker.example", False),
        ("www.okx.com", False),
        ("", False),
    ],
)
def test_is_broker_url(url, ok) -> None:
    assert is_broker_url(url, ("okx.com",)) is ok


@pytest.mark.parametrize(
    "host,ok",
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("192.168.1.20", True),
        ("10.0.0.5", True),
        ("host.docker.internal", True),
        ("opend.local", True),
        ("169.254.169.254", False),
        ("0.0.0.0", False),
        ("8.8.8.8", False),
        ("attacker.example", False),
        ("https://attacker.example", False),
        ("localhost.attacker.example", False),
    ],
)
def test_is_local_gateway_host(host, ok) -> None:
    assert is_local_gateway_host(host) is ok


def test_agent_host_override_rejects_remote_hosts() -> None:
    from src.tools.trading_connector_tool import InvalidTradingArgument, _overrides

    for bad in ("https://attacker.example", "attacker.example", "169.254.169.254"):
        with pytest.raises(InvalidTradingArgument):
            _overrides({"host": bad})
    for good in ("127.0.0.1", "opend.local", "host.docker.internal", None, ""):
        assert _overrides({"host": good})["host"] in {good, None}


def test_okx_config_refuses_foreign_host() -> None:
    from src.trading.connectors.okx.sdk import OKXConfig, OKXConfigError, build_config

    assert OKXConfig.from_mapping({}).host == "https://www.okx.com"
    for bad in ("https://attacker.example", "http://www.okx.com", "https://www.okx.com.attacker.example"):
        with pytest.raises(OKXConfigError):
            OKXConfig.from_mapping({"host": bad, "api_key": "TEST-KEY-CANARY"})
        with pytest.raises(OKXConfigError):
            build_config({"profile": "paper"}, {"host": bad})


def test_other_cloud_broker_urls_are_pinned() -> None:
    from src.trading.connectors.binance.sdk import BinanceConfig, BinanceConfigError
    from src.trading.connectors.toss.sdk import TossConfig, TossConfigError
    from src.trading.connectors.trading212.sdk import Trading212Config, Trading212ConfigError

    assert BinanceConfig.from_mapping({"testnet_host": "https://testnet.binancefuture.com"})
    with pytest.raises(BinanceConfigError):
        BinanceConfig.from_mapping({"testnet_host": "https://attacker.example"})
    assert TossConfig.from_mapping({})
    with pytest.raises(TossConfigError):
        TossConfig.from_mapping({"base_url": "http://127.0.0.1:9"})
    assert Trading212Config.from_mapping({"profile": "paper", "base_url": "https://demo.trading212.com"})
    with pytest.raises(Trading212ConfigError):
        Trading212Config.from_mapping({"base_url": "https://attacker.example"})


# --------------------------------------------------------------------------- #
# F7: a paper profile can never resolve to a live connector config
# --------------------------------------------------------------------------- #


def test_paper_profile_refuses_live_profile_override(monkeypatch, tmp_path) -> None:
    from src.trading import service
    from src.trading.connectors.alpaca import sdk as alpaca_sdk
    from src.trading.profiles import profile_by_id

    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path))
    placed: list[object] = []
    monkeypatch.setattr(
        alpaca_sdk, "place_order", lambda cfg, **kw: placed.append(cfg) or {"status": "ok"}
    )

    profile = profile_by_id("alpaca-paper-trade")
    assert service._sdk_config(profile, alpaca_sdk, {}).environment == "paper"
    # ``profile`` is no longer an override key, so the attempt is dropped before
    # it can reach the connector: the resolved config stays on the paper host
    # rather than raising. The service-level environment check still refuses a
    # config that changes environment by any other route
    # (tests/test_profile_override_isolation.py pins both layers).
    assert service._sdk_config(profile, alpaca_sdk, {"profile": "live"}).environment == "paper"
    service.place_order("AAPL", "alpaca-paper-trade", side="buy", quantity=1, profile="live")
    assert placed and all(getattr(cfg, "is_paper", False) for cfg in placed)


# --------------------------------------------------------------------------- #
# M4: child->parent paper IPC is JSON lines, never pickle
# --------------------------------------------------------------------------- #

_UNPICKLED: list[str] = []


def _mark_unpickled() -> None:
    _UNPICKLED.append("ran")


class _Payload:
    def __reduce__(self):
        return (_mark_unpickled, ())


@pytest.fixture
def sp():
    """The IPC module; its package __init__ imports defusedxml (a base dependency)."""
    pytest.importorskip("defusedxml")
    from src.tools.paper_trade import strategy_process

    return strategy_process


def _parent_conn(sp, data: bytes):
    return sp.StrategyConnection(io.BytesIO(data), io.BytesIO(), parent=True)


def test_parent_never_unpickles_child_output(sp) -> None:
    conn = _parent_conn(sp, pickle.dumps(_Payload()) + b"\n")
    assert conn.poll(5)
    with pytest.raises(EOFError):
        conn.recv()
    conn.close()
    assert _UNPICKLED == []


@pytest.mark.parametrize(
    "line",
    [b'{"tag": "ready"}\n', b"[]\n", b"[1]\n", b'["ready"]', b"not json\n"],
    ids=["object", "empty", "untagged", "no_newline", "garbage"],
)
def test_decode_child_message_rejects_malformed(sp, line) -> None:
    with pytest.raises(ValueError):
        sp.decode_child_message(line)


def test_child_message_round_trip_and_size_cap(sp, monkeypatch) -> None:
    order = {"symbol": "600519.SH", "side": "buy", "qty": 100.0, "limit": None, "reason": "x"}
    line = sp.encode_child_message(("day", "2026-01-05", [order]))
    assert sp.decode_child_message(line) == ("day", "2026-01-05", [order])
    with pytest.raises(ValueError):
        sp.encode_child_message(("day", "2026-01-05", [{"qty": float("nan")}]))

    monkeypatch.setattr(sp, "_MAX_CHILD_MESSAGE_BYTES", 32)
    conn = _parent_conn(sp, sp.encode_child_message(("day", "d", ["x" * 100])))
    assert conn.poll(5)
    with pytest.raises(EOFError):
        conn.recv()
    conn.close()


def test_malformed_child_error_is_a_protocol_failure() -> None:
    pytest.importorskip("defusedxml")
    from src.tools.paper_trade.strategy_exec import StrategyProcessError, _child_failure

    assert _child_failure(("error",), "2026-01-05").class_name == StrategyProcessError.__name__
    assert _child_failure(("error", 1, 2), "2026-01-05").class_name == StrategyProcessError.__name__
    failure = _child_failure(("error", "2026-01-06", "ZeroDivisionError"), "2026-01-05")
    assert (failure.day, failure.class_name) == ("2026-01-06", "ZeroDivisionError")


def test_paper_strategy_child_gets_scrubbed_env_and_private_cwd(sp, monkeypatch) -> None:
    pytest.importorskip("rich")
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    captured: dict = {}

    class _FakeProcess:
        def __init__(self, args, **kwargs):
            captured.update(kwargs)
            self.stdin, self.stdout = io.BytesIO(), io.BytesIO()

    monkeypatch.setattr(subprocess, "Popen", _FakeProcess)
    _, conn = sp.start_strategy_process({"source": "", "filename": "x.py", "days": (), "symbols": ()})
    workdir = Path(captured["cwd"])
    assert workdir.is_dir() and not any(workdir.iterdir())
    assert CANARY not in captured["env"].values()
    assert "OPENAI_API_KEY" not in captured["env"]
    conn.close()
    assert not workdir.exists()


# --------------------------------------------------------------------------- #
# Paper child filesystem boundary, exercised through the real child process.
# The child runs the source as given (the parent-side scanner is bypassed on
# purpose), so these pin the boundary itself, not the scanner.
# --------------------------------------------------------------------------- #

_FS_ENGINE = "class SignalEngine:\n    def generate_signals(self, ctx, trade_date):\n{body}"


def _paper_child_result(sp, body: str):
    source = _FS_ENGINE.format(body=textwrap.indent(textwrap.dedent(body), " " * 8))
    process, conn = sp.start_strategy_process(
        {"source": source, "filename": "s.py", "days": ("2026-01-05",), "symbols": ()}
    )
    try:
        assert conn.poll(60), "paper child never started"
        message = conn.recv()
        if message == ("ready",):
            conn.send((None, {}))
            assert conn.poll(60), "paper child never answered"
            message = conn.recv()
        return message
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        conn.close()


@pytest.fixture()
def secrets(tmp_path, monkeypatch):
    """A fake home secret, and a non-code file inside a sys.path entry (the agent/.env shape)."""
    home_secret = tmp_path / "home" / ".vibe-trading" / "okx.json"
    home_secret.parent.mkdir(parents=True)
    home_secret.write_text(CANARY, encoding="utf-8")
    code_root = tmp_path / "code_root"
    code_root.mkdir()
    (code_root / ".env").write_text(f"OPENAI_API_KEY={CANARY}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(code_root))
    monkeypatch.delenv("VIBE_TRADING_SANDBOX_FS", raising=False)
    return {"home": home_secret, "dotenv": code_root / ".env"}


@pytest.mark.parametrize(
    "body",
    [
        "return [open({home!r}).read()]",
        "from pathlib import Path\nreturn [getattr(Path({home!r}), 'read_text')()]",
        "return [open({dotenv!r}).read()]",
        "import os\nreturn os.listdir({home_dir!r})",
        "import os\nos.remove({home!r})\nreturn []",
        "import os\nfd = os.open({home!r}, os.O_RDONLY)\nreturn [os.read(fd, 64)]",
    ],
    ids=["open_home", "getattr_read_text", "dotenv_on_sys_path", "listdir_home", "remove_home", "os_open"],
)
def test_paper_child_cannot_reach_host_files(sp, secrets, body) -> None:
    body = body.format(home=str(secrets["home"]), dotenv=str(secrets["dotenv"]),
                       home_dir=str(secrets["home"].parent))
    message = _paper_child_result(sp, body)
    assert message == ("error", "2026-01-05", "FileAccessDenied"), message
    assert CANARY not in repr(message)
    assert secrets["home"].read_text(encoding="utf-8") == CANARY


def test_paper_child_keeps_its_own_workdir_and_lazy_imports(sp, secrets) -> None:
    body = (
        "import statistics, decimal, json\n"
        "open('scratch.txt', 'w').write('1')\n"
        "assert open('scratch.txt').read() == '1'\n"
        "return []\n"
    )
    assert _paper_child_result(sp, body) == ("day", "2026-01-05", [])


def test_required_fs_mode_refuses_without_a_mount_namespace(sp, monkeypatch) -> None:
    monkeypatch.setenv("VIBE_TRADING_SANDBOX_FS", "required")
    monkeypatch.setattr(sp.sys, "platform", "win32")
    with pytest.raises(sp.SandboxFilesystemIsolationError):
        sp.start_strategy_process({"source": "", "filename": "x.py", "days": (), "symbols": ()})


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="mount namespaces are Linux-only")
def test_fs_isolation_prefix_hides_home_runtime_root_and_dotenv(sp, tmp_path, monkeypatch) -> None:
    """Kernel layer alone (no audit hook in this child): host files are simply not there."""
    home = tmp_path / "home"
    (home / ".vibe-trading").mkdir(parents=True)
    (home / ".vibe-trading" / "okx.json").write_text(CANARY, encoding="utf-8")
    code_root = home / "app"  # code living under HOME must stay importable
    code_root.mkdir()
    (code_root / "probe_mod.py").write_text("VALUE = 7\n", encoding="utf-8")
    (code_root / ".env").write_text(CANARY, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("VIBE_TRADING_HOME", raising=False)
    monkeypatch.syspath_prepend(str(code_root))
    workdir = tmp_path / "work"
    workdir.mkdir()
    try:
        prefix = sp._fs_isolation_prefix(str(workdir))
    except sp.SandboxFilesystemIsolationError as exc:
        pytest.skip(f"host cannot create the mount namespace: {exc}")
    probe = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(code_root)!r})\n"
        "import probe_mod\n"
        f"secret = os.path.exists({str(home / '.vibe-trading' / 'okx.json')!r})\n"
        f"dotenv = open({str(code_root / '.env')!r}).read()\n"
        "open('w.txt', 'w').write('ok')\n"
        "print(json.dumps({'secret': secret, 'dotenv': dotenv, 'value': probe_mod.VALUE}))\n"
    )
    result = subprocess.run([*prefix, sys.executable, "-c", probe], cwd=workdir,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"secret": False, "dotenv": "", "value": 7}
    assert (workdir / "w.txt").read_text() == "ok"
    assert (home / ".vibe-trading" / "okx.json").read_text(encoding="utf-8") == CANARY


def test_paper_child_guard_off_is_the_control(sp, secrets, monkeypatch) -> None:
    """With the boundary off the same read succeeds, so the denials above are real."""
    monkeypatch.setenv("VIBE_TRADING_SANDBOX_FS", "off")
    body = f"open({str(secrets['home'])!r}).read()\nreturn []\n"
    assert _paper_child_result(sp, body) == ("day", "2026-01-05", [])


# --------------------------------------------------------------------------- #
# P1.1: generated strategy code runs with no network
# --------------------------------------------------------------------------- #

AGENT_ROOT = Path(__file__).resolve().parents[1]

# An audit hook cannot be removed, so anything that arms the guard must run in
# its own interpreter: arming it in-process would poison every later test.
_GUARD_PROBE = """
import json, socket, sys, threading, urllib.request

from src.core import network_guard as ng

report = {}
report["install"] = [ng.install(), ng.install()]
report["installed"] = ng.is_installed()
report["armed_before"] = ng.is_armed()

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.bind(("127.0.0.1", 0))
listener.listen(8)
port = listener.getsockname()[1]


def _serve_one():
    conn, _ = listener.accept()
    conn.close()


threading.Thread(target=_serve_one, daemon=True).start()


def _connect():
    socket.create_connection(("127.0.0.1", port), timeout=5).close()


_connect()
report["pre_block"] = "ok"


def _expect_denied(label, fn):
    try:
        fn()
    except ng.NetworkAccessDenied:
        report[label] = "denied"
    except BaseException as exc:
        report[label] = "other:" + type(exc).__name__
    else:
        report[label] = "allowed"


with ng.deny_network():
    report["armed_inside"] = ng.is_armed()
    # Allocating a socket is not the boundary; moving bytes or resolving a name is.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    report["allocate_inside"] = "ok"
    _expect_denied("connect", lambda: sock.connect(("127.0.0.1", port)))
    _expect_denied("connect_ex", lambda: sock.connect_ex(("127.0.0.1", port)))
    _expect_denied("getaddrinfo", lambda: socket.getaddrinfo("localhost", 80))
    _expect_denied("gethostbyname", lambda: socket.gethostbyname("localhost"))
    _expect_denied(
        "urlopen",
        lambda: urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=5),
    )
    _expect_denied("bind", lambda: sock.bind(("127.0.0.1", 0)))
    if sys.platform != "win32":
        # AF_UNIX can reach a host daemon, so it is denied like any other peer.
        unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _expect_denied("af_unix", lambda: unix.connect("/tmp/vibe-guard-probe.sock"))
    with ng.deny_network():
        report["nested_armed"] = ng.is_armed()
    report["armed_after_nested"] = ng.is_armed()

report["armed_after"] = ng.is_armed()
sock.close()
_connect()
report["post_block"] = "ok"
listener.close()
print(json.dumps(report))
"""

_OFF_PROBE = """
import json, socket

from src.core import network_guard as ng

report = {"install": ng.install(), "installed": ng.is_installed()}
with ng.deny_network():
    report["armed_inside"] = ng.is_armed()

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.bind(("127.0.0.1", 0))
listener.listen(8)
socket.create_connection(("127.0.0.1", listener.getsockname()[1]), timeout=5).close()
listener.close()
report["connect"] = "ok"
print(json.dumps(report))
"""


def _clean_subprocess_env(**extra: str) -> dict[str, str]:
    """Parent env minus anything that would import or configure the guard."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", network_guard.ENV_VAR}
    }
    env.update(extra)
    return env


def _run_python(
    args: list[str], *, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _run_probe(script: str, **extra_env: str) -> dict:
    result = _run_python(["-c", script], cwd=AGENT_ROOT, env=_clean_subprocess_env(**extra_env))
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("auto", "auto"),
        ("somethingelse", "auto"),
        ("off", "off"),
        ("OFF", "off"),
        ("0", "off"),
        ("false", "off"),
        ("no", "off"),
        ("required", "required"),
        ("Required", "required"),
        ("1", "required"),
        ("true", "required"),
        ("yes", "required"),
    ],
)
def test_network_mode_parsing(monkeypatch, raw: str | None, expected: str) -> None:
    if raw is None:
        monkeypatch.delenv(network_guard.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(network_guard.ENV_VAR, raw)
    assert network_guard.network_mode() == expected


def test_network_access_denied_is_not_an_oserror() -> None:
    # Retry helpers written for sockets catch OSError; a denial must not look
    # like a transient connection failure they can quietly swallow and repeat.
    assert issubclass(network_guard.NetworkAccessDenied, RuntimeError)
    assert not issubclass(network_guard.NetworkAccessDenied, OSError)


def test_deny_network_blocks_sockets_and_releases_after_the_block() -> None:
    report = _run_probe(_GUARD_PROBE)

    assert report["install"] == [True, True], "install() must be idempotent"
    assert report["installed"] is True
    assert report["armed_before"] is False, "installing must not arm the guard"
    assert report["pre_block"] == "ok", "the child's own data fetch must keep working"

    assert report["armed_inside"] is True
    assert report["allocate_inside"] == "ok"
    for label in ("connect", "connect_ex", "getaddrinfo", "gethostbyname", "urlopen", "bind"):
        assert report[label] == "denied", f"{label} reached the network: {report[label]}"
    if "af_unix" in report:
        assert report["af_unix"] == "denied"

    assert report["nested_armed"] is True
    assert report["armed_after_nested"] is True, "an inner block must not disarm the outer one"
    assert report["armed_after"] is False
    assert report["post_block"] == "ok"


def test_network_mode_off_leaves_generated_code_unguarded() -> None:
    report = _run_probe(_OFF_PROBE, **{network_guard.ENV_VAR: "off"})

    assert report["install"] is False
    assert report["installed"] is False
    assert report["armed_inside"] is False
    assert report["connect"] == "ok"


def test_sandbox_network_prefix_refuses_without_resource(monkeypatch) -> None:
    monkeypatch.setattr(sandbox_runner, "resource", None)
    with pytest.raises(sandbox_runner.SandboxNetworkIsolationError, match="no resource module"):
        sandbox_runner._sandbox_network_prefix()


def test_sandbox_network_prefix_refuses_without_unshare(monkeypatch) -> None:
    # Fake the resource module so the platform guard does not short-circuit the
    # check under test on Windows.
    monkeypatch.setattr(sandbox_runner, "resource", object())
    monkeypatch.setattr(sandbox_runner.shutil, "which", lambda _name: None)
    with pytest.raises(sandbox_runner.SandboxNetworkIsolationError, match="no 'unshare' binary"):
        sandbox_runner._sandbox_network_prefix()


def _fake_unshare(monkeypatch, works) -> list[list[str]]:
    """Install a fake ``unshare`` whose probes succeed where ``works`` says."""
    monkeypatch.setattr(sandbox_runner, "resource", object())
    monkeypatch.setattr(sandbox_runner.shutil, "which", lambda _name: "/usr/bin/unshare")
    calls: list[list[str]] = []

    class _Probe:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def _run(cmd, **_kwargs):
        calls.append(list(cmd))
        return _Probe(0 if works(cmd) else 1)

    monkeypatch.setattr(sandbox_runner.subprocess, "run", _run)
    return calls


def test_sandbox_network_prefix_refuses_when_every_probe_fails(monkeypatch) -> None:
    calls = _fake_unshare(monkeypatch, lambda _cmd: False)
    with pytest.raises(
        sandbox_runner.SandboxNetworkIsolationError, match="not usable on this host"
    ):
        sandbox_runner._sandbox_network_prefix()
    assert len(calls) == 2, "both flag sets must be probed before refusing"


def test_sandbox_network_prefix_prefers_map_root_user(monkeypatch) -> None:
    calls = _fake_unshare(monkeypatch, lambda _cmd: True)
    prefix = sandbox_runner._sandbox_network_prefix()
    assert prefix == ["/usr/bin/unshare", "--net", "--map-root-user", "--"]
    assert len(calls) == 1


def test_sandbox_network_prefix_falls_back_to_bare_net(monkeypatch) -> None:
    _fake_unshare(monkeypatch, lambda cmd: "--map-root-user" not in cmd)
    prefix = sandbox_runner._sandbox_network_prefix()
    assert prefix == ["/usr/bin/unshare", "--net", "--"]


def test_sandbox_bootstrap_installs_guard_and_forwards_argv(tmp_path: Path) -> None:
    entry = tmp_path / "entry.py"
    entry.write_text(
        "import json, sys\n"
        "from src.core.network_guard import is_installed\n"
        "print(json.dumps({'argv': sys.argv, 'installed': is_installed()}))\n",
        encoding="utf-8",
    )
    argv = sandbox_runner._sandbox_bootstrap_argv(AGENT_ROOT)
    if sandbox_runner.resource is None:
        assert argv[3:] == ["0", "0"], "no resource module: limits skipped, guard still installed"

    # cwd is NOT the agent root and PYTHONPATH is empty, so the import only
    # works because the bootstrap put agent_root on sys.path.
    result = _run_python(
        [*argv, str(entry), "--alpha", "7"],
        cwd=tmp_path,
        env=_clean_subprocess_env(),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["argv"] == [str(entry), "--alpha", "7"]
    assert report["installed"] is True


def _fake_agent_root(tmp_path: Path) -> Path:
    """A tree whose ``src.core.network_guard`` cannot be imported."""
    root = tmp_path / "fake-agent"
    core = root / "src" / "core"
    core.mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (core / "__init__.py").write_text("", encoding="utf-8")
    (core / "network_guard.py").write_text(
        "raise ImportError('guard unavailable in this build')\n", encoding="utf-8"
    )
    return root


def test_sandbox_bootstrap_fails_closed_when_guard_cannot_import(tmp_path: Path) -> None:
    root = _fake_agent_root(tmp_path)
    entry = tmp_path / "entry.py"
    entry.write_text("print('RAN')\n", encoding="utf-8")

    result = _run_python(
        [*sandbox_runner._sandbox_bootstrap_argv(root), str(entry)],
        cwd=tmp_path,
        env=_clean_subprocess_env(),
    )

    assert result.returncode == 70, "an unguarded child must not run strategy code"
    assert "sandbox network guard failed to install" in result.stderr
    assert "RAN" not in result.stdout


def test_sandbox_bootstrap_does_not_fail_closed_in_off_mode(tmp_path: Path) -> None:
    root = _fake_agent_root(tmp_path)
    entry = tmp_path / "entry.py"
    entry.write_text("print('RAN')\n", encoding="utf-8")

    result = _run_python(
        [*sandbox_runner._sandbox_bootstrap_argv(root), str(entry)],
        cwd=tmp_path,
        env=_clean_subprocess_env(**{network_guard.ENV_VAR: "off"}),
    )

    assert result.returncode == 0, result.stderr
    assert "RAN" in result.stdout


@pytest.mark.parametrize("mode", ["required", "auto"])
def test_execute_places_netns_prefix_only_when_required(
    monkeypatch, tmp_path: Path, mode: str
) -> None:
    prefix = ["/usr/bin/unshare", "--net", "--map-root-user", "--"]
    monkeypatch.setattr(sandbox_runner, "_sandbox_network_prefix", lambda: list(prefix))
    if mode == "required":
        monkeypatch.setenv(network_guard.ENV_VAR, "required")
    else:
        monkeypatch.delenv(network_guard.ENV_VAR, raising=False)

    seen: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_runner.subprocess, "run", _fake_run)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    entry = tmp_path / "entry.py"
    entry.write_text("print('x')\n", encoding="utf-8")

    sandbox_runner.Runner(timeout=60).execute(entry, run_dir, cwd=tmp_path)

    assert seen, "the sandbox subprocess was never launched"
    launch = seen[-1]
    if mode == "required":
        assert launch[: len(prefix)] == prefix
        # The namespace must wrap the python launch, not sit after it.
        assert launch[len(prefix)] == sandbox_runner.Runner(timeout=1)._pick_python_interpreter()
        assert launch[len(prefix) + 1] == "-c"
        assert launch[len(prefix) + 2] == sandbox_runner._SANDBOX_BOOTSTRAP
    else:
        assert "--" not in launch, f"auto mode must not add the namespace prefix: {launch[:4]}"
        assert launch[1] == "-c"
        assert launch[2] == sandbox_runner._SANDBOX_BOOTSTRAP
