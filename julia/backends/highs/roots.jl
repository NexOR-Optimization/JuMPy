# Compilation roots of HiGHS, registered in the image-build session after
# loading JuMPyHiGHS.
import HiGHS

Base.Experimental.entrypoint(HiGHS.Highs_destroy, (HiGHS.Optimizer,))
Base.Experimental.entrypoint(HiGHS.HighsCallbackDataIn, (Cint,))
