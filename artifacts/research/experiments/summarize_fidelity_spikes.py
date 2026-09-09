"""Preserve measured camera experiments without committing large reconstructions."""

import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    captures = root / "reconstructions"
    evidence = root / "artifacts/research/evidence"
    evidence.mkdir(exist_ok=True)
    records = {}
    for name, relative in {
        "local_camera_comparison": "local-pose-falsifier-v1/results.jsonl",
        "owning_window_comparison": "local-pose-owner-falsifier-v1/results.jsonl",
        "motion_weight_comparison": "motion-weight-falsifier-v1/results.jsonl",
        "matched_ownership_comparison": "motion-weight-falsifier-v2/results.jsonl",
        "overlap_quality": "overlap-quality-falsifier.json",
        "apple_photogrammetry": "apple-capture-80/result.json",
        "native_stream_memory_and_prefix": "indoor-travel-native-stream1000/native-stream-comparison.json",
    }.items():
        path = captures / relative
        if not path.exists():
            continue
        records[name] = {
            "source": relative,
            "result": [json.loads(line) for line in path.read_text().splitlines()]
            if path.suffix == ".jsonl"
            else json.loads(path.read_text()),
        }
    for run in ["indoor-travel-learned-cameras-v1", "indoor-travel-learned-motion10-bootstrap", "indoor-travel-native-stream1000"]:
        model = captures / run / "model"
        records[run] = {}
        for name in ["validation.json", "render-validation.json", "web-render-validation.json"]:
            path = model / name
            if path.exists():
                records[run][name] = json.loads(path.read_text())
    (evidence / "fidelity-spikes.json").write_text(json.dumps(records, indent=2) + "\n")
    for arm in ["native-window", "registered-window"]:
        source = captures / "local-pose-falsifier-v1" / f"945-{arm}.jpg"
        shutil.copy2(source, evidence / f"kitchen-{arm}.jpg")
    for run, page, name in [
        ("indoor-travel-learned-motion10-bootstrap", "05", "hybrid-app-comparison.jpg"),
        ("indoor-travel-native-stream1000", "08", "native-stream-app-comparison.jpg"),
    ]:
        source = captures / run / "model" / f"web-source-comparison-{page}.jpg"
        if source.exists():
            shutil.copy2(source, evidence / name)
    print(f"Saved {len(records)} experiment groups to {evidence}")


if __name__ == "__main__":
    main()
