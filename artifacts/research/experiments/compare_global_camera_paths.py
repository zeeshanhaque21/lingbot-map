"""Compare complete estimated camera paths using one similarity per reconstruction."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.geometry import (
    robust_similarity,
    similarity,
    transform_points,
)
from lingbot_map.reconstruction.io import digest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--baseline-cameras", type=Path, required=True)
    parser.add_argument("--candidate-cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier diagnostics; use a new output directory")
    images, _ = read_model(args.colmap_text)
    baseline = json.loads(args.baseline_cameras.read_text())
    candidate = json.loads(args.candidate_cameras.read_text())
    frame_ids = np.array([row["frame"] for row in baseline])
    if len(np.unique(frame_ids)) != len(frame_ids) or frame_ids.tolist() != [
        row["frame"] for row in candidate
    ]:
        raise ValueError("Camera paths must have identical unique frame ordering")
    if not np.array_equal(frame_ids, np.arange(len(frame_ids))):
        raise ValueError("Expected the complete consecutively numbered capture")
    reference = np.array(
        [
            np.linalg.inv(
                np.vstack([images[f"{number:06d}.jpg"]["extrinsics"], [0, 0, 0, 1]])
            )
            for number in frame_ids
        ]
    )
    train, reserved = frame_ids % 10 != 5, frame_ids % 10 == 5
    target = reference[:, :3, 3]
    # A robust path extent normalizes arbitrary monocular units, without claiming metres.
    extent = float(
        np.linalg.norm(
            np.quantile(target, 0.95, axis=0) - np.quantile(target, 0.05, axis=0)
        )
    )
    if extent <= 0 or train.sum() < 4 or not reserved.any():
        raise ValueError("Insufficient camera-path extent or reserved observations")
    _, _, principal_axes = np.linalg.svd(target - target.mean(0), full_matrices=False)
    projected_reference = (target - target.mean(0)) @ principal_axes[:2].T
    seconds = np.array([row["timestamp_seconds"] for row in baseline])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(
        *projected_reference.T, color="black", linewidth=1.5, label="COLMAP"
    )
    results = {}
    for name, records, color in [
        ("unchanged learned baseline", baseline, "#2171b5"),
        ("Cauchy camera candidate", candidate, "#d95f0e"),
    ]:
        poses = np.asarray([row["camera_to_world"] for row in records])
        fit, _ = robust_similarity(poses[train, :3, 3], target[train])
        scale = float(np.cbrt(np.linalg.det(fit[:3, :3])))
        if scale <= 0:
            raise ValueError("Camera registration has nonpositive scale")
        rotation = fit[:3, :3] / scale
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError(
                "Expected one proper similarity, without affine deformation"
            )
        aligned = transform_points(poses[:, :3, 3], fit)
        position_error = np.linalg.norm(aligned - target, axis=1)
        relative_rotation = np.einsum(
            "nij,njk->nik",
            reference[:, :3, :3].transpose(0, 2, 1),
            rotation @ poses[:, :3, :3],
        )
        angles = np.degrees(
            np.arccos(
                np.clip((np.trace(relative_rotation, axis1=1, axis2=2) - 1) / 2, -1, 1)
            )
        )
        fractions = position_error / extent
        plain_fit = similarity(poses[train, :3, 3], target[train])
        plain_errors = (
            np.linalg.norm(
                transform_points(poses[:, :3, 3], plain_fit) - target, axis=1
            )
            / extent
        )
        source_steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
        reference_steps = np.linalg.norm(np.diff(target, axis=0), axis=1)
        step_ratios = np.divide(
            reference_steps,
            source_steps,
            out=np.full_like(source_steps, np.nan),
            where=source_steps > 1e-9,
        )
        reference_turns = (
            reference[:-1, :3, :3].transpose(0, 2, 1) @ reference[1:, :3, :3]
        )
        source_turns = poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]
        turn_difference = reference_turns.transpose(0, 2, 1) @ source_turns
        turn_angles = np.degrees(
            np.arccos(
                np.clip((np.trace(turn_difference, axis1=1, axis2=2) - 1) / 2, -1, 1)
            )
        )
        results[name] = {
            "source_to_colmap_similarity": fit.tolist(),
            "scale": scale,
            "training_camera_centers": int(train.sum()),
            "reserved_camera_centers": int(reserved.sum()),
            "reserved_position_error_fraction_of_reference_extent_p50_p95_max": np.quantile(
                fractions[reserved], [0.5, 0.95, 1]
            ).tolist(),
            "reserved_orientation_difference_degrees_p50_p95_max": np.quantile(
                angles[reserved], [0.5, 0.95, 1]
            ).tolist(),
            "least_squares_control": {
                "source_to_colmap_similarity": plain_fit.tolist(),
                "reserved_position_error_fraction_of_reference_extent_p50_p95_max": np.quantile(
                    plain_errors[reserved], [0.5, 0.95, 1]
                ).tolist(),
            },
            "adjacent_rotation_difference_degrees_p50_p95_max": np.quantile(
                turn_angles, [0.5, 0.95, 1]
            ).tolist(),
            "adjacent_step_ratio_by_100_frames": [
                {
                    "first_frame": start,
                    "last_frame": min(start + 99, len(poses) - 1),
                    "median_colmap_to_source_step_ratio": float(
                        np.nanmedian(step_ratios[start : start + 99])
                    ),
                }
                for start in range(0, len(poses) - 1, 100)
            ],
            "per_frame": [
                {
                    "frame": int(number),
                    "reserved_from_similarity_fit": bool(held),
                    "position_error_colmap_units": float(error),
                    "position_error_fraction_of_reference_extent": float(fraction),
                    "orientation_difference_degrees": float(angle),
                }
                for number, held, error, fraction, angle in zip(
                    frame_ids, reserved, position_error, fractions, angles, strict=True
                )
            ],
        }
        projected = (aligned - target.mean(0)) @ principal_axes[:2].T
        axes[0, 0].plot(*projected.T, color=color, linewidth=1, label=name)
        axes[0, 1].plot(seconds, fractions * 100, color=color, linewidth=1, label=name)
        axes[1, 0].plot(seconds, angles, color=color, linewidth=1, label=name)
    axes[0, 0].set(
        title="Complete path, common principal-plane projection",
        xlabel="Principal axis 1 (COLMAP units)",
        ylabel="Principal axis 2 (COLMAP units)",
    )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(
        title="Position disagreement after one global fit",
        xlabel="Capture time (seconds)",
        ylabel="Error / robust path extent (%)",
    )
    axes[1, 0].set(
        title="Orientation disagreement",
        xlabel="Capture time (seconds)",
        ylabel="Rotation difference (degrees)",
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0,
        1,
        "Development diagnostic\n\nOne similarity fitted on 900 camera centers.\n100 centers reserved from the alignment fit.\nNo local or per-view alignment.\n\nAll camera methods used the same capture.\nAgreement does not establish truth.\nDisagreement does not identify the correct model.\nCOLMAP coordinates have no measured scale.",
        va="top",
        fontsize=12,
    )
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        axis.grid(alpha=0.2)
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, args.output / "producer.py")
    fig.savefig(args.output / "camera-path-comparison.png", dpi=180)
    plt.close(fig)
    result = {
        "source_sha256": {
            "baseline_cameras": digest(args.baseline_cameras),
            "candidate_cameras": digest(args.candidate_cameras),
            "producer": digest(Path(__file__)),
            **{
                name: digest(args.colmap_text / name)
                for name in ("cameras.txt", "images.txt", "points3D.txt")
            },
        },
        "reference_robust_path_extent_colmap_units": extent,
        "comparisons": results,
        "metric_accuracy_verified": False,
        "interpretation": "One robust similarity per entire estimated camera path, fitted only to frames not ending in 5. No local or per-view realignment and no mesh deformation. Both estimators used captured images, including reserved frames during camera estimation. This measures global cross-method consistency, not surveyed accuracy. The unchanged learned reconstruction is the do-nothing baseline for the Cauchy candidate. Differences cannot establish which camera path is correct.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {key: value for key, value in row.items() if key != "per_frame"}
                for name, row in results.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
