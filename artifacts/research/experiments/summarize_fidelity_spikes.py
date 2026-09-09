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
        "remaining_local_views": "remaining-views-falsifier-v1/results.jsonl",
        "fresh_context_bfloat16": "local-depth-reset-760/local-comparison.jsonl",
        "fresh_context_float32": "local-depth-reset-760-float32/local-comparison.jsonl",
        "visibility_carving": "visibility-carving-v1/results.json",
        "camera_pair_fits": "pair-pose-falsifier-v1/results.jsonl",
        "joint_pose_graph": "joint-pose-graph-v1/results.json",
        "direct_bundle_huber": "direct-bundle-v1/results.json",
        "direct_bundle_huber_objective": "direct-bundle-v1/objective-diagnostics.json",
        "direct_bundle_cauchy": "direct-bundle-cauchy-v1/results.json",
        "mapanything_smoke": "mapanything-mac-smoke-785/completed.json",
        "mapanything_local": "mapanything-mac-760/completed.json",
        "mapanything_local_comparison": "mapanything-mac-760/local-comparison.jsonl",
        "photometric_synthetic_mps": "photometric-depth-synthetic-mps.json",
        "brush_chair_comparison": "brush-chair-760/evaluation-final/results.json",
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
    for run in ["joint-pose-graph-v1", "direct-bundle-v1", "direct-bundle-cauchy-v1"]:
        path = captures / run / "camera-comparison.json"
        if path.exists():
            comparison = json.loads(path.read_text())
            for value in comparison.values():
                if isinstance(value, dict):
                    value.pop("per_frame", None)
            records[f"{run}_camera_comparison"] = comparison
    for run in [
        "mapanything-window-images-760",
        "mapanything-calibration-760",
        "mapanything-posed-760",
        "mapanything-fixed-cameras-760",
    ]:
        records[run] = {}
        for relative in [
            "completed.json",
            "anchoring.json",
            "local-comparison.jsonl",
            "model/validation.json",
            "model/render-validation.json",
            "model/web-render-validation.json",
        ]:
            path = captures / run / relative
            if path.exists():
                records[run][relative] = (
                    [json.loads(line) for line in path.read_text().splitlines()]
                    if path.suffix == ".jsonl"
                    else json.loads(path.read_text())
                )
    for run in [
        "photometric-depth-760",
        "photometric-depth-760-spatial",
        "photometric-depth-760-no-occlusion",
    ]:
        records[run] = {}
        for relative in [
            "signature.json",
            "trace.jsonl",
            "rejection-diagnostics.json",
            "baseline-results.json",
            "refined-results.json",
            "baseline/model/render-validation.json",
            "baseline/model/web-render-validation.json",
            "refined/model/render-validation.json",
            "refined/model/web-render-validation.json",
        ]:
            path = captures / run / relative
            if path.exists():
                records[run][relative] = (
                    [json.loads(line) for line in path.read_text().splitlines()]
                    if path.suffix == ".jsonl"
                    else json.loads(path.read_text())
                )
    for run in [
        "indoor-travel-learned-cameras-v1",
        "indoor-travel-learned-motion10-bootstrap",
        "indoor-travel-native-stream1000",
    ]:
        model = captures / run / "model"
        records[run] = {}
        for name in [
            "validation.json",
            "render-validation.json",
            "web-render-validation.json",
        ]:
            path = model / name
            if path.exists():
                records[run][name] = json.loads(path.read_text())
    (evidence / "fidelity-spikes.json").write_text(json.dumps(records, indent=2) + "\n")
    for arm in ["native-window", "registered-window"]:
        source = captures / "local-pose-falsifier-v1" / f"945-{arm}.jpg"
        shutil.copy2(source, evidence / f"kitchen-{arm}.jpg")
    source = captures / "mapanything-calibration-760" / "785-mapanything-window.jpg"
    if source.exists():
        shutil.copy2(source, evidence / "mapanything-chair-comparison.jpg")
    for run, page, name in [
        ("indoor-travel-learned-motion10-bootstrap", "05", "hybrid-app-comparison.jpg"),
        ("indoor-travel-native-stream1000", "08", "native-stream-app-comparison.jpg"),
        (
            "direct-bundle-cauchy-v1/candidate",
            "13",
            "robust-bundle-chair-comparison.jpg",
        ),
        (
            "photometric-depth-760-spatial/refined",
            "00",
            "photometric-spatial-chair-comparison.jpg",
        ),
    ]:
        source = captures / run / "model" / f"web-source-comparison-{page}.jpg"
        if source.exists():
            shutil.copy2(source, evidence / name)
    for source in (captures / "brush-chair-760/evaluation-final").glob(
        "*-comparison.jpg"
    ):
        shutil.copy2(source, evidence / f"brush-{source.name}")
    print(f"Saved {len(records)} experiment groups to {evidence}")


if __name__ == "__main__":
    main()
