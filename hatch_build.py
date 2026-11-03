"""
Tag the wheel for its platform when it contains the compiled backends that
`julia/build_backend.py --package src` copied into `src/jumpy/_backends`.
Without them, the wheel stays pure Python and only the JuliaCall interface
works.
"""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        backends = Path(self.root, "src", "jumpy", "_backends")
        tags = {path.read_text().strip() for path in backends.glob("*/wheel-tag")}
        if self.target_name != "wheel" or not tags:
            return
        if len(tags) > 1:
            raise RuntimeError(f"The backends were built for different platforms: {tags}")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{tags.pop()}"
