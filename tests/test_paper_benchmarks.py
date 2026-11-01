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
from jumpy.model import Model
from mock_ops import MockOps


class PaperBenchmarkTests(unittest.TestCase):
    def csv_fixture(self, path, **overrides):
        rows = [dict(model="lqcp", size=100, frontend=frontend, backend="highs",
                     variables=10301, median_seconds=seconds, **overrides)
                for frontend, seconds in zip(FRONTENDS, (0.4, 0.3, 0.1, 0.105))]
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
