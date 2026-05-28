import re
from pathlib import Path

def extract_miss_level(
    path: Path,
) -> int | None:
    """Extract the integer missing level from a directory 'miss_{miss_level}."""
    match = re.search(r"miss_(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None