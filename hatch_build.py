"""
Tag the wheel for its platform when it contains the compiled backend that
`julia/package_bundle.py` copied into `src/jumpy/_backend`. Without it, the
wheel stays pure Python and only the JuliaCall interface works.
"""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        tag_file = Path(self.root, "src", "jumpy", "_backend", "wheel-tag")
        if self.target_name != "wheel" or not tag_file.exists():
            return
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{tag_file.read_text().strip()}"
