"""Dependency-free checks for the paper ports and batched variable bounds."""

from pathlib import Path
import csv
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "paper"))
from models import facility, lqcp
from report import FRONTENDS, load_results, markdown, mermaid, write_report
from run import select_frontends, summarize
from run_jumpy import construct
from types import SimpleNamespace
from jumpy.model import Model
from mock_ops import MockOps


class PaperBenchmarkTests(unittest.TestCase):
    def csv_fixture(self, path, **overrides):
        rows = [dict(model="lqcp", size=100, frontend=frontend, backend="highs",
                     variables=10301, median_seconds=seconds)
                for frontend, seconds in zip(FRONTENDS, (0.4, 0.3, 0.1, 0.105))]
        for row in rows:
            row.update(overrides)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def test_csv_report_orders_columns_and_compares_python_to_grouped_julia(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            self.csv_fixture(path)
            groups = load_results(path)
            table = markdown(groups)
            self.assertIn("JuMP cached | JuMP direct | JuMP direct + GenOpt | JuMPy.HiGHS", table)
            self.assertIn("10,301 | 400.0 ms | 300.0 ms | **100.0 ms** | 105.0 ms | 1.05×", table)
            chart = mermaid(groups)
            self.assertIn("xychart-beta", chart)
            self.assertIn("bar [400.000, 300.000, 100.000, 105.000]", chart)
            output, chart_path = Path(directory) / "table.md", Path(directory) / "charts.md"
            self.assertEqual(write_report(path, output, chart_path), table)
            self.assertEqual(output.read_text(), table)
            self.assertEqual(chart_path.read_text(), chart)

    def test_report_rejects_invalid_or_duplicate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            for invalid in ("nan", "inf", "-1", "0"):
                self.csv_fixture(path)
                text = path.read_text().replace(",0.4", f",{invalid}", 1)
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "Invalid median"):
                    load_results(path)
            self.csv_fixture(path)
            with path.open("a") as stream:
                stream.write(path.read_text().splitlines()[1] + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_results(path)

    def test_report_includes_all_seven_frontends_and_supports_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            rows = self.csv_fixture(path)
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                for frontend, seconds in (("highspy", .05), ("pyomo", .6), ("linopy", .2)):
                    writer.writerow(dict(rows[0], frontend=frontend, median_seconds=seconds))
            groups = load_results(path)
            table = markdown(groups)
            self.assertIn("JuMPy.HiGHS | highspy | Pyomo | Linopy", table)
            self.assertIn("**50.0 ms** | 600.0 ms | 200.0 ms | 1.05×", table)
            self.assertEqual(len({line.count("|") for line in table.splitlines()}), 1)
            self.assertIn('"highspy", "Pyomo", "Linopy"', mermaid(groups))
            subset = {key: {f: row for f, row in group.items() if f == "linopy"}
                      for key, group in groups.items()}
            self.assertIn("Variables | Linopy |", markdown(subset))
            self.assertNotIn("JuMPy.HiGHS", markdown(subset))
            self.assertNotIn("grouped Julia", markdown(subset))

    def test_backend_selects_supported_frontends(self):
        self.assertEqual(select_frontends("highs"), list(FRONTENDS))
        self.assertEqual(select_frontends("nexor"), list(FRONTENDS)[:4])
        self.assertEqual(select_frontends("highs", ["linopy"]), ["linopy"])
        with self.assertRaisesRegex(ValueError, "require --backend highs"):
            select_frontends("nexor", ["pyomo"])

    def test_nexor_constructs_locally_and_reports_its_backend(self):
        ops = MockOps()
        model = Model(ops)
        self.assertIs(construct(SimpleNamespace(Model=lambda: model), "nexor", "facility", 3), model)
        self.assertEqual(ops.optimize_calls, 0)
        self.assertEqual(model.num_variables(), 199)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            self.csv_fixture(path, backend="nexor")
            self.assertIn("JuMPy.NexOR", markdown(load_results(path)))

    def test_mof_timings_are_summarized_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [dict(event="sample", model="lqcp", size=3, frontend="jumpy",
                            backend="nexor", variables=19, seconds=build,
                            export_seconds=export, peak_rss_bytes=10)
                       for build, export in ((1, 3), (2, 4), (3, 5))]
            build = summarize(records, Path(directory))
            export = summarize(records, Path(directory), "export_seconds", "mof-summary.csv")
            self.assertEqual(build[0]["median_seconds"], 2)
            self.assertEqual(export[0]["median_seconds"], 4)

    def test_bounds_are_one_backend_call_per_set(self):
        class BulkOps(MockOps):
            def __init__(self):
                super().__init__()
                self.blocks = []

            def add_variable_constraints(self, variables, set_):
                self.blocks.append((variables, set_))

            def apply(self, *args):
                raise AssertionError("Bounds must not index each variable in Python")

        ops = BulkOps()
        model = Model(ops)
        variables = model.variables(1_000_000, lower=0, upper=1, binary=True)
        self.assertEqual(ops.blocks, [(variables.ref, set_) for set_ in (
            ("GreaterThan", 0.0), ("LessThan", 1.0), ("ZeroOne",),
        )])

    def test_facility_uses_five_constraint_templates(self):
        ops = MockOps()
        model = Model(ops)
        facility(model, 3)
        self.assertEqual(sum(array[2] for array in ops.arrays), 199)
        # Variable bounds plus five template constraints, independent of n^3.
        self.assertEqual(len(ops.constraints), 54 + 6 + 48 + 5)

    def test_lqcp_uses_four_constraint_templates(self):
        ops = MockOps()
        model = Model(ops)
        lqcp(model, 3)
        self.assertEqual(sum(array[2] for array in ops.arrays), 19)
        self.assertEqual(len(ops.constraints), 2 * 19 + 4)
        self.assertEqual(model.objective.sense, "min")

    def test_invalid_sizes(self):
        model = Model(MockOps())
        with self.assertRaises(ValueError):
            facility(model, 0)
        with self.assertRaises(ValueError):
            lqcp(model, 2)


if __name__ == "__main__":
    unittest.main()
