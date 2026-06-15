import yaml
import shutil
from pathlib import Path
from utils.paths import get_project_root

class ConfigWriter:
    @staticmethod
    def snapshot_config(
        cfg: dict,
        type_cfg: str,
        out_dir: Path,
    ) -> None:
        cfg_path = Path(out_dir / f"{type_cfg}.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

    @staticmethod
    def _resolve_config_path(relative_path: str, config_parent: Path) -> Path:
        """
        Resolve a config path with fallback strategies.
        
        Tries:
        1. Resolve relative to config file's parent directory
        2. If not found and path contains 'configs/', try from project root
        3. If not found, try going up more directory levels
        """
        v_path = Path(relative_path)
        
        # Try 1: Direct relative resolution
        candidate = (config_parent / v_path).resolve()
        if candidate.exists():
            return candidate
        
        # Try 2: If path references 'configs' subdirectory, try from project root
        if "configs" in str(v_path):
            project_root = get_project_root()
            # Try resolving as absolute within configs structure
            if str(v_path).startswith("../"):
                # Remove leading ../ and try from project root
                normalized = str(v_path).lstrip("./")
                while normalized.startswith("../"):
                    normalized = normalized[3:]
                candidate = (project_root / normalized).resolve()
                if candidate.exists():
                    return candidate
            
            # Try direct concatenation with project root
            candidate = (project_root / v_path).resolve()
            if candidate.exists():
                return candidate
        
        # Try 3: Go up additional levels from config parent
        for _ in range(1, 6):
            candidate = (config_parent / ("../" * _) / v_path).resolve()
            if candidate.exists():
                return candidate
        
        # If nothing found, return the first attempt
        return (config_parent / v_path).resolve()

    @staticmethod
    def snapshot_config_tree(
        config_path: Path,
        out_dir: Path,
        visited=None,
    ) -> None:
        """
        Recursively copy referenced YAML config files into cfg directory.
        """
        if visited is None:
            visited = set()

        config_path = config_path.resolve()
        if config_path in visited:
            return

        visited.add(config_path)
        
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}")
            return
        
        shutil.copy2(config_path, out_dir)

        # Load YAML to find nested config references
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        def walk(obj, base_dir) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str) and "cfg" in k and isinstance(v, str):
                        nested_path = ConfigWriter._resolve_config_path(v, base_dir)
                        ConfigWriter.snapshot_config_tree(config_path=nested_path, out_dir=out_dir, visited=visited)
                    else:
                        walk(v, base_dir)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item, base_dir)

        walk(data, config_path.parent)