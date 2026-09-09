"""Audit saved loop matches against the actual exported camera poses."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from .io import digest


def audit_closure(source, cameras):
    root = Path(source).parent
    required = [
        root / name
        for name in (
            "closure-pairs.txt",
            "closure-matching-completion.json",
            "pipeline-completion.json",
            "colmap/database.db",
        )
    ]
    if not all(path.exists() for path in required):
        return {"status": "not_audited", "navigation_closure": False}
    pairs_path, matching, pipeline, database = required
    if any(
        json.loads(path.read_text())["returncode"] != 0 for path in (matching, pipeline)
    ):
        return {"status": "failed_stage", "navigation_closure": False}
    by_frame = {camera["frame"]: camera for camera in cameras}
    records = []
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        ids = {name: image_id for image_id, name in names.items()}
        for pair in pairs_path.read_text().splitlines():
            a, b = pair.split()
            ia, ib = sorted((ids[a], ids[b]))
            frames = [int(Path(names[i]).stem) for i in (ia, ib)]
            if any(frame not in by_frame for frame in frames):
                continue
            row = db.execute(
                "SELECT rows, data, config FROM two_view_geometries WHERE pair_id=?",
                (ia * 2147483647 + ib,),
            ).fetchone()
            if row is None or row[0] < 15 or row[2] not in (2, 3):
                continue
            matches = np.frombuffer(row[1], np.uint32).reshape(-1, 2)
            pixels = []
            for column, image_id in enumerate((ia, ib)):
                n, cols, data = db.execute(
                    "SELECT rows, cols, data FROM keypoints WHERE image_id=?",
                    (image_id,),
                ).fetchone()
                xy = np.frombuffer(data, np.float32).reshape(n, cols)[
                    matches[:, column], :2
                ]
                # COLMAP centers start at 0.5; these exported cameras use integer centers.
                offset = 0.5 - by_frame[frames[column]].get("pixel_center_offset", 0.0)
                pixels.append(np.c_[xy - offset, np.ones(len(xy))])
            ca, cb = [by_frame[frame] for frame in frames]
            relative = np.linalg.inv(cb["camera_to_world"]) @ ca["camera_to_world"]
            t = relative[:3, 3]
            if np.linalg.norm(t) < 1e-8:
                continue  # Coincident poses cannot verify translation with an essential matrix.
            skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
            f = (
                np.linalg.inv(cb["intrinsics"]).T
                @ skew
                @ relative[:3, :3]
                @ np.linalg.inv(ca["intrinsics"])
            )
            l2, l1 = pixels[0] @ f.T, pixels[1] @ f
            residual = np.abs(np.sum(pixels[1] * l2, axis=1)) / np.maximum(
                np.sqrt((l2[:, :2] ** 2 + l1[:, :2] ** 2).sum(axis=1)), 1e-12
            )
            records.append(
                {
                    "frames": frames,
                    "inliers": len(matches),
                    "median_sampson_pixels": float(np.median(residual)),
                }
            )
    supported = [r for r in records if r["median_sampson_pixels"] <= 3]
    first = min(by_frame)
    last = max(by_frame)
    endpoint_supported = any(
        min(r["frames"]) == first and max(r["frames"]) == last for r in supported
    )
    return {
        "status": "evaluated",
        "database_sha256": digest(database),
        "pairs_sha256": digest(pairs_path),
        "verified_pairs": len(records),
        "pairs_with_median_under_3_pixels": len(supported),
        "navigation_closure": endpoint_supported and len(supported) >= 3,
        "median_pair_sampson_pixels": float(
            np.median([r["median_sampson_pixels"] for r in records])
        )
        if records
        else None,
        "interpretation": "Training-match consistency, not independent accuracy. Endpoints need not coincide. Navigation closure requires supported endpoint matches and at least three supported return pairs.",
        "pairs": records,
    }
