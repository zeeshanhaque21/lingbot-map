"""Audit disputed camera turns using retained pixel correspondences and final calibration."""

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.reconstruction.colmap_io import read_model
from lingbot_map.reconstruction.io import digest


def rotation_degrees(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def essential_from_pose(pose):
    t = pose[:3, 3]
    t = t / np.linalg.norm(t)
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return skew @ pose[:3, :3]


def sampson_pixels(essential, first, second, focal):
    x = np.column_stack([first, np.ones(len(first))])
    y = np.column_stack([second, np.ones(len(second))])
    ex, ety = x @ essential.T, y @ essential
    denominator = np.sum(ex[:, :2] ** 2, axis=1) + np.sum(ety[:, :2] ** 2, axis=1)
    return (
        np.abs(np.sum(y * ex, axis=1)) / np.sqrt(np.maximum(denominator, 1e-20)) * focal
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--colmap-text", type=Path, required=True)
    parser.add_argument("--baseline-cameras", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Captured frame pair, e.g. 493:494",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier diagnostics; choose a new output directory")
    images, _ = read_model(args.colmap_text)
    baseline = {
        row["frame"]: np.array(row["camera_to_world"])
        for row in json.loads(args.baseline_cameras.read_text())
    }
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, args.output / "producer.py")
    rows = []
    with (args.output / "pairs.jsonl").open("x") as trace:
        for pair in args.pair:
            first, second = map(int, pair.split(":"))
            names = [f"{number:06d}.jpg" for number in (first, second)]
            identifiers = [
                connection.execute(
                    "SELECT image_id FROM images WHERE name=?", (name,)
                ).fetchone()[0]
                for name in names
            ]
            row = connection.execute(
                "SELECT rows, cols, data, config FROM two_view_geometries WHERE pair_id=?",
                (min(identifiers) * 2147483647 + max(identifiers),),
            ).fetchone()
            result = {
                "first_frame": first,
                "second_frame": second,
                "verified_matches": 0 if row is None else row[0],
                "database_config": None if row is None else row[3],
            }
            if row is not None and row[0] >= 30:
                matches = np.frombuffer(row[2], np.uint32).reshape(row[0], row[1])
                if identifiers[0] > identifiers[1]:
                    matches = matches[:, ::-1]
                pixels, normalized, poses, focal = [], [], [], []
                for index, (identifier, name) in enumerate(
                    zip(identifiers, names, strict=True)
                ):
                    count, columns, blob = connection.execute(
                        "SELECT rows, cols, data FROM keypoints WHERE image_id=?",
                        (identifier,),
                    ).fetchone()
                    points = (
                        np.frombuffer(blob, np.float32)
                        .reshape(count, columns)[matches[:, index], :2]
                        .astype(np.float64)
                    )
                    camera = images[name]["camera"]
                    k = camera["intrinsics"]
                    pixels.append(points)
                    normalized.append(
                        cv2.undistortPoints(points[:, None], k, None)[:, 0]
                    )
                    poses.append(
                        np.linalg.inv(
                            np.vstack([images[name]["extrinsics"], [0, 0, 0, 1]])
                        )
                    )
                    focal.extend([k[0, 0], k[1, 1]])
                focal = float(np.mean(focal))
                reserved = np.arange(len(matches)) % 5 == 0
                train = ~reserved
                cv2.setRNGSeed(20260909)
                essential, mask = cv2.findEssentialMat(
                    normalized[0][train],
                    normalized[1][train],
                    np.eye(3),
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1 / focal,
                )
                if essential is None:
                    result["status"] = "essential_estimation_failed"
                else:
                    hypotheses = []
                    for candidate in essential.reshape(-1, 3, 3):
                        count, rotation, translation, positive = cv2.recoverPose(
                            candidate,
                            normalized[0][train],
                            normalized[1][train],
                            np.eye(3),
                            mask=mask.copy(),
                        )
                        hypotheses.append(
                            (count, candidate, rotation, translation, positive)
                        )
                    count, essential, rotation, translation, positive = max(
                        hypotheses, key=lambda item: item[0]
                    )
                    global_pose = np.linalg.inv(poses[1]) @ poses[0]
                    baseline_pose = np.linalg.inv(baseline[second]) @ baseline[first]
                    _, planar_mask = cv2.findHomography(
                        pixels[0][train], pixels[1][train], cv2.RANSAC, 2
                    )
                    rays0 = np.column_stack(
                        [normalized[0][train], np.ones(train.sum())]
                    )
                    rays1 = (
                        np.column_stack([normalized[1][train], np.ones(train.sum())])
                        @ rotation
                    )
                    cosine = np.sum(rays0 * rays1, axis=1) / (
                        np.linalg.norm(rays0, axis=1) * np.linalg.norm(rays1, axis=1)
                    )
                    parallax = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
                    positive = positive.ravel() != 0
                    errors = {
                        "fresh_pair_essential": sampson_pixels(
                            essential,
                            normalized[0][reserved],
                            normalized[1][reserved],
                            focal,
                        ),
                        "unchanged_global_colmap": sampson_pixels(
                            essential_from_pose(global_pose),
                            normalized[0][reserved],
                            normalized[1][reserved],
                            focal,
                        ),
                    }
                    result.update(
                        {
                            "status": "evaluated",
                            "fit_correspondences": int(train.sum()),
                            "reserved_correspondences": int(reserved.sum()),
                            "fit_essential_inliers": int(np.count_nonzero(mask)),
                            "fit_positive_depth_inliers": int(count),
                            "homography_fit_inlier_fraction": None
                            if planar_mask is None
                            else float(planar_mask.mean()),
                            "median_positive_depth_parallax_degrees": float(
                                np.median(parallax[positive])
                            )
                            if positive.any()
                            else None,
                            "fresh_pair_turn_degrees": rotation_degrees(rotation),
                            "global_colmap_turn_degrees": rotation_degrees(
                                global_pose[:3, :3]
                            ),
                            "learned_baseline_turn_degrees": rotation_degrees(
                                baseline_pose[:3, :3]
                            ),
                            "pair_to_global_rotation_difference_degrees": rotation_degrees(
                                rotation.T @ global_pose[:3, :3]
                            ),
                            "pair_to_learned_rotation_difference_degrees": rotation_degrees(
                                rotation.T @ baseline_pose[:3, :3]
                            ),
                            "reserved_sampson_error_pixels_p50_p95": {
                                name: np.quantile(error, [0.5, 0.95]).tolist()
                                for name, error in errors.items()
                            },
                            "fresh_pair_rotation": rotation.tolist(),
                            "fresh_pair_translation_direction": translation.ravel().tolist(),
                        }
                    )
                    np.savez_compressed(
                        args.output / f"{first:06d}-{second:06d}.npz",
                        first_pixels=pixels[0],
                        second_pixels=pixels[1],
                        first_normalized=normalized[0],
                        second_normalized=normalized[1],
                        reserved=reserved,
                        essential=essential,
                        global_pose=global_pose,
                        learned_pose=baseline_pose,
                    )
            else:
                result["status"] = "insufficient_verified_correspondences"
            trace.write(json.dumps(result) + "\n")
            trace.flush()
            rows.append(result)
            print(json.dumps(result), flush=True)
    connection.close()
    result = {
        "pairs": rows,
        "source_sha256": {
            "database": digest(args.database),
            "baseline_cameras": digest(args.baseline_cameras),
            "producer": digest(Path(__file__)),
            **{
                name: digest(args.colmap_text / name)
                for name in ("cameras.txt", "images.txt", "points3D.txt")
            },
        },
        "metric_accuracy_verified": False,
        "interpretation": "Read-only reprojection audit using retained verified COLMAP correspondences, with fresh essential matrices fitted using final camera calibration and 20% of pair correspondences reserved from this new fit. Camera estimation previously used the same images and database, so this is a development consistency test, not independent truth. Rotation magnitudes and differences also compare the unchanged learned baseline. High planar support or low triangulation angle may make two-view pose ambiguous. No camera path or mesh is modified.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
