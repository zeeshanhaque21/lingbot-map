"""Copy immutable local run evidence into the research artifact directory."""

import argparse
import json
import shutil
from pathlib import Path

from lingbot_map.reconstruction.io import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for name, root in [("sample", args.sample), ("full", args.full)]:
        model = root / "model"
        evidence[name] = {
            key: json.loads((model / file).read_text())
            for key, file in [
                ("mesh", "validation.json"),
                ("source_view_check", "render-validation.json"),
                ("app_view_check", "web-render-validation.json"),
                ("app_export", "app-export.json"),
            ]
        }
        evidence[name]["asset_bytes"] = (model / "property-detail.glb").stat().st_size
    for root, source, destination in [
        (args.sample, "web-source-comparison-00.jpg", "sample-app-comparison.jpg"),
        (args.full, "web-source-comparison-03.jpg", "full-app-comparison.jpg"),
    ]:
        source, destination = root / "model" / source, args.output / destination
        if destination.exists() and digest(destination) != digest(source):
            raise ValueError(
                "Existing evidence differs; preserve it and select a new output"
            )
        if not destination.exists():
            shutil.copyfile(source, destination)
    write_json(args.output / "measurements.json", evidence)
    print("status: complete")
    print("output: " + json.dumps(str(args.output.resolve())))


if __name__ == "__main__":
    main()
