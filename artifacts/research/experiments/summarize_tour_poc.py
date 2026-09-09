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
    refinements = []
    for root in sorted(args.tour.glob("fixanything-*")):
        if not (root / "started.json").exists():
            continue
        started = json.loads((root / "started.json").read_text())
        completion = (
            json.loads((root / "completion.json").read_text())
            if (root / "completion.json").exists()
            else None
        )
        entry = {"name": root.name, "started": started, "completion": completion}
        for name, filename in (
            ("stitch", "stitch/stitch.json"),
            ("comparison", "evaluation/comparison.json"),
            ("source_comparison", "source-comparison/comparison.json"),
        ):
            if (root / filename).exists():
                entry[name] = json.loads((root / filename).read_text())
        refinements.append(entry)
    full_completed = any(
        r["completion"]
        and r["completion"].get("success")
        and r["completion"].get("full_spherical_sweep")
        for r in refinements
    )
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
                        "refined_panorama",
                        "refined_panorama_sha256",
                        "refinement_review",
                    )
                    if k in s
                }
                for s in stations
            ],
            "closure_baseline": {k: v for k, v in baseline.items() if k != "pairs"},
            "baseline_cameras_sha256": digest(baseline_cameras),
            "closure_candidate": {k: v for k, v in candidate.items() if k != "pairs"},
            "raw_stitch": json.loads(
                (args.tour / "sweep-000-v1/raw-stitch/stitch.json").read_text()
            ),
            "refinements": refinements,
            "source_view_baseline": (
                json.loads(
                    (args.tour / "source-comparison-baseline-v1/comparison.json").read_text()
                )
                if (args.tour / "source-comparison-baseline-v1/comparison.json").exists()
                else None
            ),
            "full_spherical_inference_completed": full_completed,
            "limitations": [
                "Training-match residuals are not independent accuracy.",
                "Spherical directional coverage is not observed surface coverage.",
                "Generated details require structural review against the capture.",
                "An absent completion record does not establish process liveness.",
            ],
        },
    )


if __name__ == "__main__":
    main()
