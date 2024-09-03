from pathlib import Path


def posix_to_pypath(path: Path) -> str:
    posix_path = path.as_posix()
    pypath = posix_path.replace("/", ".").strip(".")
    return pypath
