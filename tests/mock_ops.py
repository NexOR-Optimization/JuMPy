"""
A mock ops object recording native operations as tuples, so tests can check
exactly what a backend receives — pure Python, no Julia needed.
"""

from types import SimpleNamespace


class MockOps:
    def __init__(self):
        self.constraints = []
        self.arrays = []
        self.value_calls = []
        self.value_result = 1.25
        self.optimize_calls = 0
        self.dual_result = -3.0
        self.dual_calls = []
        self.status = 1
        self.primal_available = True
        self.dual_available = True
        self.objective_result = 2.5
        self.MOI = SimpleNamespace(
            LessThan=lambda upper: ("LessThan", upper),
            GreaterThan=lambda lower: ("GreaterThan", lower),
            EqualTo=lambda value: ("EqualTo", value),
            ZeroOne=lambda: ("ZeroOne",),
            Integer=lambda: ("Integer",),
        )

    def constant(self, v):
        return v

    def apply(self, op, args):
        return (op, *args)

    def iterator(self, values):
        return ("iterator", tuple(values))

    def float_array(self, values):
        return ("data", tuple(values))

    def add_variables(self, count):
        ref = ("array", len(self.arrays), count)
        self.arrays.append(ref)
        return ref

    def add_constraint(self, func, set_):
        self.constraints.append((func, set_))
        return ("constraint", len(self.constraints))

    def optimize(self):
        self.optimize_calls += 1
        return self.status

    def termination_status(self):
        return self.status

    def has_values(self):
        return self.primal_available

    def has_duals(self):
        return self.dual_available

    def objective_value(self):
        return self.objective_result

    def dual(self, constraint):
        self.dual_calls.append(constraint)
        return self.dual_result

    def value(self, func):
        self.value_calls.append(func)
        return self.value_result

    def free(self):
        pass
