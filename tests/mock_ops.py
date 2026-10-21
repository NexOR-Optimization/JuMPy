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

    def optimize(self):
        self.optimize_calls += 1
        return 1

    def value(self, func):
        self.value_calls.append(func)
        return self.value_result

    def free(self):
        pass
