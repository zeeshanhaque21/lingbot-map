"""Regenerate compact tour evidence from the saved run artifacts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lingbot_map.reconstruction.closure import audit_closure
from lingbot_map.reconstruction.io import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tour", type=Path, default=Path("reconstructions/tours/loop-v1")
    )
    parser.add_argument(
        "--baseline", type=Path, default=Path("reconstructions/examples/loop/final")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/research/evidence/property-tour.json"),
    )
    args = parser.parse_args()
    tour = json.loads((args.tour / "tour.json").read_text())
    stations = json.loads((args.tour / "stations.json").read_text())
    baseline_cameras = args.baseline / "model/cameras.json"
    baseline = audit_closure(tour["source"], json.loads(baseline_cameras.read_text()))
    candidate = json.loads((args.tour / "closure-audit.json").read_text())
    write_json(
        args.output,
        {
            "tour": tour,
            "stations": [
                {
                    k: s[k]
                    for k in (
                        "id",
                        "frame",
                        "observed_sphere_fraction",
                        "generated",
                        "panorama_sha256",
                    )
                }
                for s in stations
            ],
            "closure_baseline": {k: v for k, v in baseline.items() if k != "pairs"},
            "baseline_cameras_sha256": digest(baseline_cameras),
            "closure_candidate": {k: v for k, v in candidate.items() if k != "pairs"},
            "raw_stitch": json.loads(
                (args.tour / "sweep-000-v1/raw-stitch/stitch.json").read_text()
            ),
            "limitations": [
                "Training-match residuals are not independent accuracy.",
                "Spherical directional coverage is not observed surface coverage.",
                "FixAnything inference has not yet completed.",
            ],
        },
    )


if __name__ == "__main__":
    main()
