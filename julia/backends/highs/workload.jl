# Traced in ordinary Julia before building the image; see julia/README.md.
# Image generation does not initialize the solver JLL, so the image build does
# not run it.
import JuMPy
import JuMPyHiGHS

JuMPy.Client.workload(
    JuMPyHiGHS;
    attributes = (
        "presolve" => "on",
        "mip_detect_symmetry" => true,
        "random_seed" => 1,
        "mip_rel_gap" => 1e-4,
    ),
)
