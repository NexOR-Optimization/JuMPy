"""Check the Python LQCP ports against a JuMP MOF export, including all coefficients."""

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "paper"))
from python_models import BUILDERS, construct
from validate import canonical

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("highspy", "pyomo", "linopy", "numpy", "scipy"))


def native_canonical(model):
    """Use the same representation as validate.py, undoing Pyomo's column order."""
    import highspy
    import numpy as np
    from scipy import sparse

    data = model.highs.getModel()
    lp, hessian = data.lp_, data.hessian_
    nv = lp.num_col_
    order = np.arange(nv)
    if model.solver is not None:
        variables = list(model.model.y.values()) + list(model.model.u.values())
        order = np.array([model.solver._pyomo_var_to_solver_var_map[id(v)] for v in variables])
    original = np.argsort(order)

    def number(value):
        return round(float(value), 12)

    def terms(indices, values):
        return tuple(sorted((int(original[i]), number(v)) for i, v in zip(indices, values) if number(v)))

    matrix = lp.a_matrix_
    matrix_type = sparse.csr_matrix if matrix.format_ == highspy.MatrixFormat.kRowwise else sparse.csc_matrix
    a = matrix_type((matrix.value_, matrix.index_, matrix.start_), shape=(lp.num_row_, nv)).tocsr()
    a.sum_duplicates()
    rows = Counter()
    for i in range(lp.num_row_):
        # All LQCP rows are equalities with zero constant.
        assert lp.row_lower_[i] == lp.row_upper_[i]
        row = a.getrow(i)
        rows[((0.0, terms(row.indices, row.data), ()),
              ("EqualTo", (("value", number(lp.row_lower_[i])),)))] += 1
    for i in range(nv):
        variable = (0.0, ((int(original[i]), 1.0),), ())
        rows[(variable, ("GreaterThan", (("lower", number(lp.col_lower_[i])),)))] += 1
        rows[(variable, ("LessThan", (("upper", number(lp.col_upper_[i])),)))] += 1
    quadratic = []
    for col in range(nv):
        for k in range(hessian.start_[col], hessian.start_[col + 1]):
            key = tuple(sorted((int(original[col]), int(original[hessian.index_[k]]))))
            if number(hessian.value_[k]):
                quadratic.append((key, number(hessian.value_[k])))
    objective = (number(lp.offset_), terms(range(nv), lp.col_cost_), tuple(sorted(quadratic)))
    assert lp.sense_ == highspy.ObjSense.kMinimize
    return nv, "min", objective, rows


@unittest.skipUnless(AVAILABLE, "Install benchmarks/paper/requirements.txt")
class PythonPaperModelTests(unittest.TestCase):
    def test_all_coefficients_match_jump(self):
        # Exported from the grouped JuMP formulation in benchmarks/paper/models.jl,
        # checked against the original paper's JuMP formulation by validate.py.
        expected = canonical(ROOT / "tests" / "fixtures" / "lqcp3.mof.json")
        for frontend in BUILDERS:
            with self.subTest(frontend=frontend):
                model = construct(frontend, 3)
                try:
                    self.assertEqual(native_canonical(model), expected)
                    self.assertEqual(model.highs.getNumCol(), 19)
                    self.assertEqual(model.highs.getNumRow(), 16)
                    self.assertEqual(model.highs.getOptionValue("threads")[1], 1)
                    self.assertEqual(model.highs.getOptionValue("presolve")[1], "off")
                    self.assertEqual(model.highs.getOptionValue("time_limit")[1], 0)
                    self.assertFalse(model.highs.getLp().col_names_)
                finally:
                    model.close()

    def test_larger_grid_and_solution_agree(self):
        import highspy
        reference = construct("highspy", 5, solve=True)
        try:
            expected = native_canonical(reference)
            self.assertEqual(reference.highs.getModelStatus(), highspy.HighsModelStatus.kOptimal)
            objective = reference.highs.getObjectiveValue()
            for frontend in ("pyomo", "linopy"):
                with self.subTest(frontend=frontend):
                    model = construct(frontend, 5, solve=True)
                    try:
                        self.assertEqual(native_canonical(model), expected)
                        self.assertEqual(model.highs.getModelStatus(), highspy.HighsModelStatus.kOptimal)
                        self.assertAlmostEqual(model.highs.getObjectiveValue(), objective, places=9)
                    finally:
                        model.close()
        finally:
            reference.close()

    def test_invalid_grid(self):
        for frontend in BUILDERS:
            with self.assertRaisesRegex(ValueError, "at least three"):
                construct(frontend, 2)


if __name__ == "__main__":
    unittest.main()
