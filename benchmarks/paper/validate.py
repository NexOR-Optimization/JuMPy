"""Compare MOF exports up to row order and floating-point rounding."""

from collections import Counter
import json
from pathlib import Path


def canonical(path):
    data = json.loads(Path(path).read_text())
    variables = {variable["name"]: i for i, variable in enumerate(data["variables"])}

    def number(value):
        return round(float(value), 12)

    def polynomial(function):
        kind = function["type"]
        if kind == "Variable":
            return (0.0, ((variables[function["name"]], 1.0),), ())
        if kind not in ("ScalarAffineFunction", "ScalarQuadraticFunction"):
            raise ValueError(f"Unexpected function type: {kind}")
        linear = Counter()
        for term in function.get("terms", function.get("affine_terms", [])):
            linear[variables[term["variable"]]] += term["coefficient"]
        quadratic = Counter()
        for term in function.get("quadratic_terms", []):
            key = tuple(sorted((variables[term["variable_1"]], variables[term["variable_2"]])))
            quadratic[key] += term["coefficient"]
        return (
            number(function.get("constant", 0)),
            tuple(sorted((index, number(value)) for index, value in linear.items() if number(value))),
            tuple(sorted((index, number(value)) for index, value in quadratic.items() if number(value))),
        )

    rows = Counter()
    for constraint in data["constraints"]:
        set_ = constraint["set"]
        set_key = (set_["type"], tuple(sorted((k, number(v)) for k, v in set_.items() if k != "type")))
        rows[(polynomial(constraint["function"]), set_key)] += 1
    objective = data["objective"]
    return len(variables), objective["sense"], polynomial(objective["function"]), rows


def compare(reference, candidate):
    left, right = canonical(reference), canonical(candidate)
    if left[:3] != right[:3] or left[3] != right[3]:
        missing, extra = left[3] - right[3], right[3] - left[3]
        raise AssertionError({"header_reference": left[:3], "header_candidate": right[:3],
                              "missing_rows": list(missing.items())[:3], "extra_rows": list(extra.items())[:3]})
    return {"variables": left[0], "constraints": sum(left[3].values()), "equivalent": True}


if __name__ == "__main__":
    import sys
    print(json.dumps(compare(*sys.argv[1:])))
