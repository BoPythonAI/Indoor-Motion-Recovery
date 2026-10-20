"""Expand asset paths and validate the files required by a configuration."""
import argparse
import json
import os
import re
from pathlib import Path


def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        result = os.path.expandvars(value)
        if re.search(r"\$\{[^}]+\}", result):
            raise ValueError("Unresolved environment variable: " + result)
        return result
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    config = expand(json.loads(args.template.read_text(encoding="utf-8")))
    config["student_checkpoint"] = str(args.checkpoint.resolve())
    config["output"] = str(args.output_dir.resolve())
    required = ["student_checkpoint", "tracker_checkpoint", "terrain_checkpoint",
                "motion_bank", "motion_metadata", "surface_mjcf_path",
                "reference_plan_manifest", "reference_original_bank"]
    missing = [config[k] for k in required if k in config and not Path(config[k]).is_file()]
    if not Path(config["scene_dir"]).is_dir():
        missing.append(config["scene_dir"])
    if missing:
        raise FileNotFoundError("Required external assets are missing:\n" + "\n".join(missing))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a new empty evaluation output directory")
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite an existing local configuration")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("Prepared:", args.out)


if __name__ == "__main__":
    main()
