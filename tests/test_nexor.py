"""Compiled NexOR client against a local implementation of the wire protocol.

Run with JUMPY_BACKEND=nexor after building libjumpy_nexor. No API key or
production server is needed.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jumpy import backend
from jumpy.nexor import Model, minimize, sin

pytestmark = pytest.mark.skipif(
    os.environ.get("JUMPY_BACKEND") != "nexor" or backend.find_lib("nexor") is None,
    reason="requires JUMPY_BACKEND=nexor and libjumpy_nexor",
)


@pytest.fixture
def remote(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer test-key":
                self.reply(401, {"error": {"code": "unauthorized", "message": "bad key"}})
                return
            if self.path == "/solve/v1/problems":
                length = int(self.headers["Content-Length"])
                self.server.submission = json.loads(self.rfile.read(length))
                self.reply(200, {"problem_id": "test", "status": "queued"})
            elif self.path == "/solve/v1/problems/test/wait?timeout_seconds=60":
                self.reply(200, {"problem_id": "test", "status": "finished"})
            else:
                self.reply(404, {})

        def do_GET(self):
            if self.path != "/solve/v1/problems/test/result?inline=1":
                self.reply(404, {})
                return
            self.reply(200, {"solution": {
                "attributes": {
                    "termination_status": "OPTIMAL",
                    "primal_status": "FEASIBLE_POINT",
                    "dual_status": "NO_SOLUTION",
                    "result_count": 1,
                    "objective_value": 6.0,
                    "raw_status": "test solution",
                },
                "primal": [1.0, 2.0, 3.0],
            }})

        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    monkeypatch.setenv("NEXOR_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("NEXOR_API_KEY", "test-key")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_compiled_nexor_requires_a_solver(remote):
    m = Model()
    try:
        m.variable(lower=0)
        with pytest.raises(RuntimeError):
            m.optimize()
        assert not hasattr(remote, "submission")
    finally:
        m.close()


def test_compiled_nexor_solve(remote):
    m = Model()
    try:
        m.set_attribute("solver", "HiGHS")
        m.set_attribute("presolve", "off")
        x = m.variables(3, lower=0)
        demand = m.parameter([1, 2, 3])
        i = m.iterator(range(3))
        m.constraint_group(x[i] >= demand[i])
        m.objective = minimize(sum(x))
        m.optimize()
        assert [m.value(v) for v in x] == [1.0, 2.0, 3.0]
        assert m.has_values()
        assert m.objective_value() == 6.0
        submission = remote.submission
        assert submission["solver"] == "highs-mof"
        assert submission["options"]["solver_options"]["highs-mof"]["silent"]
        assert submission["options"]["solver_options"]["highs-mof"]["presolve"] == "off"
        assert len(submission["problem"]["variables"]) == 3
        assert len(submission["problem"]["constraints"]) == 6
        # Editing the direct model must invalidate its solution and submit again.
        m.constraint(sin(x[0]) <= 1)
        assert not m.has_values()
        m.optimize()
        assert remote.submission["problem"]["has_scalar_nonlinear"]
        assert [m.value(v) for v in x] == [1.0, 2.0, 3.0]
    finally:
        m.close()
