"""
A mock ops object recording native operations as tuples, so tests can check
exactly what a backend receives — pure Python, no Julia needed.
"""

from types import SimpleNamespace


class MockOps:
    def __init__(self):
        self.constraints = []
        self.num_vars = 0
        self.MOI = SimpleNamespace(
            LessThan=lambda upper: ("LessThan", upper),
            GreaterThan=lambda lower: ("GreaterThan", lower),
            EqualTo=lambda value: ("EqualTo", value),
            ZeroOne=lambda: ("ZeroOne",),
            Integer=lambda: ("Integer",),
        )

    def constant(self, v):
        return v

    def variable(self, index):
        return ("var", index)

    def apply(self, op, args):
        return (op, *args)

    def iterator(self, values):
        return ("iterator", tuple(values))

    def contiguous_variables(self, start, count):
        return ("block", start, count)

    def float_array(self, values):
        return ("data", tuple(values))

    def add_variables(self, count):
        start = self.num_vars
        self.num_vars += count
        return start

    def add_constraint(self, func, set_):
        self.constraints.append((func, set_))

    def free(self):
        pass
