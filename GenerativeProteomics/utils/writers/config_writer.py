import yaml
import shutil
from pathlib import Path

class ConfigWriter:
    def snapshot_config(
        cls,
        cfg: dict,
        type_cfg: str,
        out_dir: Path,
    ) -> None:
        cfg_path = Path(out_dir / f"{type_cfg}.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

    def snapshot_config_tree(
        cls,
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
        
        shutil.copy2(config_path, out_dir)

        # Load YAML to find nested config references
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        def walk(obj, base_dir) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str) and "config" in k and isinstance(v, str):
                        v_path = Path(v)
                        if v_path.is_absolute():
                            nested_path = v_path
                        else:
                            # If path already starts with the same dir name, avoid duplication
                            if v_path.parts[0] == base_dir.name:
                                nested_path = (base_dir.parent / v_path).resolve()
                            else:
                                nested_path = (base_dir / v_path).resolve()

                        cls.snapshot_config_tree(config_path=nested_path, out_dir=out_dir, visited=visited)
                    else:
                        walk(v, base_dir)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item, base_dir)

        walk(data, config_path.parent)