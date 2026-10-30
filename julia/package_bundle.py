"""
Copy a JuliaC bundle into the Python package so that hatchling ships it in a
platform wheel:

    python julia/package_bundle.py julia/build src/jumpy/_backend

Wheels cannot store symlinks, so the aliases of a library are copies. To keep
the wheel small, only the aliases that can be loaded are kept: the install name
(SONAME on Linux) and the names that a bundled binary refers to, such as the
`lib/libhighs.so` path that JuliaC stores for a JLL library.
Static libraries, headers, executables and man pages of the artifacts are
dropped. The platform tag of the wheel is written to `<dest>/wheel-tag`, which
`hatch_build.py` reads.
"""

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys

# Directories and suffixes of the bundle that the compiled library never loads.
SKIP_DIRS = {"bin", "include", "man", "pkgconfig"}
SKIP_SUFFIXES = (".a", ".la", ".h", ".trace")
# Oldest glibc of manylinux2014, the oldest tag that pip still widely supports.
MIN_GLIBC = (2, 17)


def _run(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def install_name(path):
    """The name the dynamic loader uses to find this library, or None."""
    if sys.platform == "darwin":
        lines = _run("otool", "-D", path).splitlines()
        return os.path.basename(lines[1]) if len(lines) > 1 else None
    match = re.search(r"\(SONAME\).*\[(.*)\]", _run("readelf", "-d", path))
    return match.group(1) if match else None


# A library extension and its version, such as `.so.1.13`. The name before it
# is read backwards from the match: a regular expression starting with the name
# backtracks over every run of name characters in the binary.
EXTENSION = re.compile(rb"\.(?:so|dylib)(?:\.\d+)*")
NAME_CHARS = re.compile(rb"[\w+.-]+$")


def referenced_names(paths):
    """The library names, such as `libopenlibm.so` in `julia/libopenlibm.so:`,
    appearing in the binaries at `paths`."""
    names = set()
    for path in paths:
        with open(path, "rb") as io:
            data = io.read()
        for match in EXTENSION.finditer(data):
            start = max(0, match.start() - 128)
            stem = NAME_CHARS.search(data[start : match.start()])
            if stem is not None:
                names.add((stem.group() + match.group()).decode())
    return names


def _digest(path):
    with open(path, "rb") as io:
        return hashlib.sha256(io.read()).hexdigest()


def collect(bundle):
    """Map each kept path relative to `bundle` to its source path."""
    groups = {}
    for root, dirs, files in os.walk(bundle):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(SKIP_SUFFIXES):
                continue
            path = os.path.join(root, name)
            groups.setdefault((root, _digest(path)), []).append(path)
    binaries = [paths[0] for paths in groups.values() if _is_library(paths[0])]
    referenced = referenced_names(binaries)
    kept = {}
    for paths in groups.values():
        if len(paths) > 1:
            names = referenced | {install_name(paths[0])}
            loaded = [p for p in paths if os.path.basename(p) in names]
            # Copies of a file that is not a library are all kept.
            paths = loaded or paths
        for path in paths:
            kept[os.path.relpath(path, bundle)] = path
    return kept


def _linux_tag(paths):
    glibc = MIN_GLIBC
    for path in paths:
        for major, minor in re.findall(
            r"GLIBC_(\d+)\.(\d+)", _run("readelf", "-W", "--dyn-syms", path)
        ):
            glibc = max(glibc, (int(major), int(minor)))
    return f"manylinux_{glibc[0]}_{glibc[1]}_{platform.machine()}"


def _macos_tag(paths):
    minimum = (10, 9)
    # The deployment target is `minos` in LC_BUILD_VERSION and `version` in
    # LC_VERSION_MIN_MACOSX. Other load commands, such as LC_SOURCE_VERSION,
    # also have a `version` field.
    fields = {"LC_BUILD_VERSION": "minos", "LC_VERSION_MIN_MACOSX": "version"}
    for path in paths:
        field = None
        for line in _run("otool", "-l", path).splitlines():
            words = line.split()
            if words[:1] == ["cmd"]:
                field = fields.get(words[1])
            elif field is not None and words[:1] == [field]:
                found = re.match(r"(\d+)\.(\d+)", words[1])
                minimum = max(minimum, (int(found[1]), int(found[2])))
    # Since macOS 11, wheel tags only use the major version.
    major, minor = minimum if minimum[0] < 11 else (minimum[0], 0)
    return f"macosx_{major}_{minor}_{platform.machine()}"


def _is_library(path):
    return re.search(r"\.(so|dylib)(\.|$)", path) is not None


def platform_tag(paths):
    libraries = [p for p in paths if _is_library(p)]
    if sys.platform == "darwin":
        return _macos_tag(libraries)
    if sys.platform.startswith("linux"):
        return _linux_tag(libraries)
    raise NotImplementedError(f"No wheel tag for {sys.platform}")


def main(bundle, dest):
    kept = collect(bundle)
    shutil.rmtree(dest, ignore_errors=True)
    for relpath, source in kept.items():
        target = os.path.join(dest, relpath)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)
    tag = platform_tag(list(kept.values()))
    with open(os.path.join(dest, "wheel-tag"), "w") as io:
        io.write(tag + "\n")
    size = sum(os.path.getsize(p) for p in kept.values())
    print(f"Copied {len(kept)} files ({size / 1e6:.0f} MB) to {dest} for {tag}")


if __name__ == "__main__":
    main(*sys.argv[1:])
