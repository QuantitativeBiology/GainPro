from pathlib import Path

def get_project_root(marker="requirements.txt") -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")