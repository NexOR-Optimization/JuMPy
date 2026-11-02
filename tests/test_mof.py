"""Local model export through the ordinary JuMPy API."""

import importlib
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jumpy import backend, bridge_juliacall
from jumpy.model import Model
from mock_ops import MockOps


class MofTests(unittest.TestCase):
    def test_local_model_api_accepts_paths_and_rejects_closed_models(self):
        ops = MockOps()
        m = Model(ops)
        m.variables(3)
        self.assertEqual(m.num_variables(), 3)
        path = Path("modèle.mof.json")
        m.write_mof(path)
        self.assertEqual(ops.exported, str(path))
        self.assertEqual(ops.optimize_calls, 0)
        m.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            m.num_variables()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            m.write_mof(path)

    def test_compiled_export_checks_native_errors(self):
        class Library:
            count = 2
            status = 0

            def jumpy_new_model(self):
                return 123

            def jumpy_free_model(self, model):
                return 0

            def jumpy_num_variables(self, model):
                assert model == 123
                return self.count

            def jumpy_write_mof(self, model, path):
                assert model == 123
                self.path = path
                return self.status

        lib = Library()
        ops = backend.JuliacOps(lib, None)
        self.assertEqual(ops.num_variables(), 2)
        ops.write_mof("modèle.mof.json")
        self.assertEqual(lib.path, "modèle.mof.json".encode())
        lib.count = -1
        with self.assertRaisesRegex(RuntimeError, "variable count"):
            ops.num_variables()
        lib.status = -1
        with self.assertRaisesRegex(RuntimeError, "export model"):
            ops.write_mof("missing/model.mof.json")
        ops.free()

    def test_old_libraries_are_rejected_before_initialization(self):
        for missing in ("jumpy_num_variables", "jumpy_write_mof"):
            class Library:
                def __getattr__(self, name):
                    if name == missing:
                        raise AttributeError(name)
                    if name in ("jl_init_with_image_handle", "jl_is_initialized"):
                        raise AssertionError("An incompatible library must not initialize Julia")
                    return lambda *args: None

            with patch.dict(sys.modules, {"juliacall": None}), patch.object(backend.ctypes, "CDLL", return_value=Library()):
                with self.assertRaisesRegex(RuntimeError, "Rebuild"):
                    backend._init_lib("old-library.so")

    def test_juliacall_export_uses_shared_helper(self):
        calls = []
        ops = object.__new__(bridge_juliacall.JuliaCallOps)
        ops._model = object()
        ops._jump = SimpleNamespace(
            num_variables=lambda model: 3,
            write_mof=lambda model, path: calls.append((model, path)),
        )
        self.assertEqual(ops.num_variables(), 3)
        ops.write_mof("local.mof.json")
        self.assertEqual(calls, [(ops._model, "local.mof.json")])

    @unittest.skipUnless(os.environ.get("JUMPY_BACKEND") in ("juliac", "nexor"),
                         "requires a normal compiled backend")
    def test_compiled_mof_export_is_local(self):
        name = "nexor" if os.environ["JUMPY_BACKEND"] == "nexor" else "highs"
        jp = importlib.import_module(f"jumpy.{name}")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "NEXOR_API_KEY": "", "NEXOR_SERVER_URL": "http://127.0.0.1:1",
        }):
            m = jp.Model()
            try:
                x = m.variables(3, lower=0, upper=4)
                i = m.iterator(range(3))
                m.constraint_group(x[i] <= 4)
                m.objective = jp.minimize(x[0] ** 2 + 3 * x[2])
                path = Path(directory) / "model.mof.json"
                m.write_mof(path)
                data = json.loads(path.read_text())
                self.assertEqual(m.num_variables(), 3)
                self.assertEqual(len(data["variables"]), 3)
                self.assertEqual(len(data["constraints"]), 9)
                self.assertEqual(data["objective"]["function"]["type"], "ScalarQuadraticFunction")
                self.assertFalse(m.has_values())
            finally:
                m.close()


if __name__ == "__main__":
    unittest.main()
