"""Whole-module strategy scanner (P0.3): bypass corpus and legitimate-code corpus.

Each bypass reached a forbidden capability (shell, env secrets, network,
module object graph, file I/O outside the run dir) past the old
reachable-code-only scanner. Payloads are only parsed, never executed.
"""

from __future__ import annotations

import uuid

import pytest

from backtest.runner import _load_module_from_file


def _load(tmp_path, source: str):
    path = tmp_path / "signal_engine.py"
    path.write_text('"""Generated signal engine."""\n' + source, encoding="utf-8")
    return _load_module_from_file(path, f"se_{uuid.uuid4().hex}")


_ENGINE = "class SignalEngine:\n    def generate(self, data_map):\n"

_BYPASSES = {
    # Reachability gaps
    "helper_class_method": (
        "import os\n"
        "class _H:\n    def go(self):\n        return os.system('id')\n"
        + _ENGINE + "        return _H().go()\n"
    ),
    "function_passed_by_reference": (
        "import os\n"
        "def _h(x):\n    return os.system('id')\n"
        + _ENGINE + "        return {k: v.apply(_h) for k, v in data_map.items()}\n"
    ),
    "function_alias_variable": (
        "import os\n"
        "def _h():\n    return os.system('id')\n"
        + _ENGINE + "        f = _h\n        return f()\n"
    ),
    "unused_toplevel_requests": "import requests\n" + _ENGINE + "        return {}\n",
    # os renamed or passed around
    "import_os_as": "import os as o\n" + _ENGINE + "        return o.system('id')\n",
    "os_assigned_to_variable": "import os\n" + _ENGINE + "        m = os\n        return m.system('id')\n",
    "os_passed_to_function": "import os\n" + _ENGINE + "        return getter(os)\n",
    # Modules reached through other objects
    "pandas_reexported_os": "import pandas as pd\n" + _ENGINE + "        return pd.io.common.os.environ\n",
    "os_sys_modules": "import os\n" + _ENGINE + "        return os.sys.modules\n",
    "os_dunder_dict": "import os\n" + _ENGINE + "        return os.__dict__['system']('id')\n",
    "operator_attrgetter": (
        "import os\nimport operator\n" + _ENGINE + "        return operator.attrgetter('system')(os)\n"
    ),
    "inspect_frames": "import inspect\n" + _ENGINE + "        return inspect.currentframe()\n",
    "generator_frame_globals": (
        _ENGINE + "        g = (x for x in ())\n        return g.gi_frame.f_back.f_globals\n"
    ),
    # Process control
    "os_execv": "import os\n" + _ENGINE + "        return os.execv('/bin/sh', ['sh'])\n",
    "os_posix_spawn": "import os\n" + _ENGINE + "        return os.posix_spawn('/bin/sh', ['sh'], {})\n",
    "os_kill": "import os\n" + _ENGINE + "        return os.kill(1, 9)\n",
    "asyncio_import": "import asyncio\n" + _ENGINE + "        return {}\n",
    # Data libraries fetching on their own
    "yfinance_import": "import yfinance as yf\n" + _ENGINE + "        return yf.download('AAPL')\n",
    "ccxt_import": "import ccxt\n" + _ENGINE + "        return {}\n",
    "pandas_url_read": (
        "import pandas as pd\n" + _ENGINE + "        return pd.read_csv('https://attacker.example/x.csv')\n"
    ),
    "pandas_absolute_read": "import pandas as pd\n" + _ENGINE + "        return pd.read_csv('/etc/passwd')\n",
    "pandas_variable_read": "import pandas as pd\n" + _ENGINE + "        p = 'x'\n        return pd.read_json(p)\n",
    "pandas_read_sql": "import pandas as pd\n" + _ENGINE + "        return pd.read_sql('select 1', None)\n",
    "numpy_absolute_load": "import numpy as np\n" + _ENGINE + "        return np.load('/tmp/x.npy')\n",
    "to_csv_with_path": _ENGINE + "        data_map['a'].to_csv('/tmp/x.csv')\n        return {}\n",
    "to_pickle_relative": _ENGINE + "        data_map['a'].to_pickle('x.pkl')\n        return {}\n",
    "numpy_save": "import numpy as np\n" + _ENGINE + "        np.save('x.npy', 1)\n        return {}\n",
    "open_url_literal": _ENGINE + "        return open('https://attacker.example')\n",
    # Indirect data-I/O calls the attribute check cannot see
    "from_pandas_import_read_csv": (
        "from pandas import read_csv\n" + _ENGINE + "        return read_csv('https://attacker.example/x')\n"
    ),
    "read_csv_as_value": (
        "import pandas as pd\n" + _ENGINE + "        f = pd.read_csv\n        return f('/etc/passwd')\n"
    ),
    "getattr_literal_read_csv": (
        "import pandas as pd\n" + _ENGINE + "        return getattr(pd, 'read_csv')('/etc/passwd')\n"
    ),
    "getattr_computed_on_module": (
        "import pandas as pd\n" + _ENGINE + "        return getattr(pd, 'read_' + 'csv')('/etc/passwd')\n"
    ),
    "getattr_pathlib_read_text": (
        "from pathlib import Path\n" + _ENGINE
        + "        return getattr(Path.home() / 'secret.txt', 'read_text')()\n"
    ),
    "numpy_load_allow_pickle_positional": (
        "import numpy as np\n" + _ENGINE + "        return np.load('x.npy', None, True)\n"
    ),
}

# Every one of these executes a pickle payload sitting in the run dir; a
# relative path is no protection. Exercised end to end below.
_DESERIALIZERS = {
    "pd_read_pickle": "import pandas as pd\n" + _ENGINE + "        return pd.read_pickle('payload.pkl')\n",
    "pd_io_read_pickle": (
        "import pandas as pd\n" + _ENGINE + "        return pd.io.pickle.read_pickle('payload.pkl')\n"
    ),
    "from_import_alias": (
        "from pandas import read_pickle as rp\n" + _ENGINE + "        return rp('payload.pkl')\n"
    ),
    "getattr_alias": (
        "import pandas as pd\n" + _ENGINE + "        return getattr(pd, 'read_pickle')('payload.pkl')\n"
    ),
    "value_alias": (
        "import pandas as pd\n" + _ENGINE + "        f = pd.read_pickle\n        return f('payload.pkl')\n"
    ),
    "np_load_allow_pickle": (
        "import numpy as np\n" + _ENGINE + "        return np.load('payload.pkl', allow_pickle=True)\n"
    ),
    "pd_read_hdf": "import pandas as pd\n" + _ENGINE + "        return pd.read_hdf('payload.h5')\n",
    "pd_hdfstore": "import pandas as pd\n" + _ENGINE + "        return pd.HDFStore('payload.h5')['x']\n",
    "joblib_load": "import joblib\n" + _ENGINE + "        return joblib.load('payload.pkl')\n",
    "dill_load": "import dill\n" + _ENGINE + "        return dill.load(open('payload.pkl', 'rb'))\n",
    "cloudpickle_load": "import cloudpickle\n" + _ENGINE + "        return cloudpickle.load(open('payload.pkl', 'rb'))\n",
    "underscore_pickle": "import _pickle\n" + _ENGINE + "        return _pickle.load(open('payload.pkl', 'rb'))\n",
    "shelve_open": "import shelve\n" + _ENGINE + "        return shelve.open('payload')['x']\n",
    "yaml_unsafe": "import yaml\n" + _ENGINE + "        return yaml.unsafe_load(open('payload.yml'))\n",
}
_BYPASSES.update({f"deserializer_{k}": v for k, v in _DESERIALIZERS.items()})


class _Payload:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


_CHILD = (
    "import sys\n"
    "from pathlib import Path\n"
    "from backtest.runner import _load_module_from_file\n"
    "try:\n"
    "    m = _load_module_from_file(Path('signal_engine.py'), 'signal_engine')\n"
    "except ValueError as exc:\n"
    "    print('REJECTED', exc)\n"
    "    sys.exit(0)\n"
    "m.SignalEngine().generate({})\n"
    "print('RAN')\n"
)


def _run_child(run_dir, agent_root):
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(agent_root)}
    return subprocess.run(
        [sys.executable, "-c", _CHILD], cwd=run_dir, env=env, capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize("source", list(_DESERIALIZERS.values()), ids=list(_DESERIALIZERS))
def test_deserializer_payload_never_executes_in_the_child(tmp_path, source) -> None:
    import pickle
    from pathlib import Path

    agent_root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "outside" / "PWNED"
    marker.parent.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = pickle.dumps(_Payload(str(marker)))
    for name in ("payload.pkl", "payload.h5", "payload.yml", "payload"):
        (run_dir / name).write_bytes(payload)
    (run_dir / "signal_engine.py").write_text('"""Generated signal engine."""\n' + source, encoding="utf-8")

    result = _run_child(run_dir, agent_root)

    assert "REJECTED" in result.stdout, result.stdout + result.stderr
    assert not marker.exists()


def test_deserializer_payload_is_live(tmp_path) -> None:
    """Control: the same payload really runs when something unpickles it."""
    import pickle

    marker = tmp_path / "PWNED"
    pickle.loads(pickle.dumps(_Payload(str(marker)))).close()
    assert marker.exists()


@pytest.mark.parametrize("source", list(_BYPASSES.values()), ids=list(_BYPASSES))
def test_bypass_is_rejected(tmp_path, source) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _load(tmp_path, source)


_LEGITIMATE = {
    "relative_csv_read": "import pandas as pd\n" + _ENGINE + "        return pd.read_csv('data/prices.csv')\n",
    "to_json_returns_string": _ENGINE + "        return {k: v.to_json() for k, v in data_map.items()}\n",
    "to_csv_without_target": _ENGINE + "        s = data_map['a'].to_csv(index=False)\n        return {}\n",
    "os_path_join": "import os\n" + _ENGINE + "        return os.path.join('a', 'b')\n",
    "numpy_math": "import numpy as np\n" + _ENGINE + "        return np.where(np.zeros(3) > 0, 1, -1)\n",
    "clean_helper_class": (
        "class _Helper:\n    def score(self, df):\n        return df['close'].pct_change()\n"
        + _ENGINE + "        return {k: _Helper().score(v) for k, v in data_map.items()}\n"
    ),
    "lambda_apply": _ENGINE + "        return {k: v['close'].apply(lambda x: x * 2) for k, v in data_map.items()}\n",
    "signal_attribute_names": (
        _ENGINE + "        self.signal = 1\n        out = {}\n"
        "        for k, v in data_map.items():\n            out[k] = v.assign(signal=self.signal)['signal']\n"
        "        return out\n"
    ),
    "json_loads": "import json\n" + _ENGINE + "        return json.loads('{}')\n",
    "typing_and_math": "import math\nfrom typing import Dict\n" + _ENGINE + "        return {'x': math.sqrt(4)}\n",
    # In-memory parsing touches no file
    "numpy_loadtxt_stringio": (
        "import numpy as np\nfrom io import StringIO\n" + _ENGINE + "        return np.loadtxt(StringIO('1 2 3'))\n"
    ),
    "pandas_read_csv_io_stringio": (
        "import io\nimport pandas as pd\n" + _ENGINE + "        return pd.read_csv(io.StringIO('a\\n1'))\n"
    ),
    "numpy_load_bytesio_no_pickle": (
        "import io\nimport numpy as np\n" + _ENGINE + "        return np.load(io.BytesIO(b''), allow_pickle=False)\n"
    ),
    "getattr_on_user_object": _ENGINE + "        return {k: getattr(v, 'close') for k, v in data_map.items()}\n",
}


@pytest.mark.parametrize("source", list(_LEGITIMATE.values()), ids=list(_LEGITIMATE))
def test_legitimate_strategy_code_still_loads(tmp_path, source) -> None:
    assert hasattr(_load(tmp_path, source), "SignalEngine")
