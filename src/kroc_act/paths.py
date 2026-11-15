"""Keep project reads, writes and cleanup inside this checkout; reject symlinks."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must remain inside {PROJECT_ROOT}") from error
    current = PROJECT_ROOT
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {current}")
    return path


def atomic_json(path, data):
    import json
    path = project_path(path)
    temporary = project_path(path.with_name(path.name + ".tmp"))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
