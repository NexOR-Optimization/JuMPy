"""
A mock ops object recording every MOI call as a tuple, so tests can check
exactly what a backend receives — pure Python, no Julia needed.
"""


class MockOps:
    def __init__(self):
        self.constraints = []
        self.groups = []
        self.num_vars = 0

    def constant(self, v):
        return v

    def variable(self, index):
        return ("var", index)

    def scalar_nonlinear(self, head, args):
        return (head, *args)

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

    def add_constraint(self, func, sense, rhs):
        self.constraints.append((func, sense, rhs))

    def add_constraint_group(self, func, sense, linear):
        self.groups.append((func, sense, linear))

    def free(self):
        pass
