"""Dependency-free checks for the paper ports and batched variable bounds."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "paper"))
from models import facility, lqcp
from jumpy.model import Model
from mock_ops import MockOps


class PaperBenchmarkTests(unittest.TestCase):
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
