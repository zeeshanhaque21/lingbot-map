"""Falsifier: fixed global rotations plus measured feature tracks and learned depth.

Standalone experiment intentionally does not import the pipeline implementation.
Usage: python depth_pose_spike.py SOURCE COLMAP_TEXT OUTPUT
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation


source, model, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=False)
camera_rows = [x.split() for x in (model / "cameras.txt").read_text().splitlines() if x and not x.startswith("#")]
cameras = {}
for row in camera_rows:
    fx, fy, cx, cy = map(float, row[4:])
    cameras[int(row[0])] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
images = {}
lines = iter((model / "images.txt").read_text().splitlines())
for line in lines:
    if not line or line.startswith("#"):
        continue
    row = line.split()
    observations = np.array(next(lines).split(), float).reshape(-1, 3)
    if not Path(row[9]).stem.isdecimal():
        continue
    q = np.array(row[1:5], float)
    r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    images[int(Path(row[9]).stem)] = (r, np.array(row[5:8], float), cameras[int(row[8])], observations)

files = sorted((source / "windows").glob("*.npz"))
tracks = defaultdict(list)
scales = []
previous = None
scale = 1.
frame_depths = {}
for chunk, file in enumerate(files):
    data = dict(np.load(file))
    if previous is not None:
        _, ii, jj = np.intersect1d(previous["frame_ids"], data["frame_ids"], return_indices=True)
        x = previous["depth"][ii, ::8, ::8]
        y = data["depth"][jj, ::8, ::8]
        ratio = np.median(x / y)
        scale *= ratio
        print("overlap", chunk, float(ratio), float(np.median(abs(x - y * ratio) / x)), flush=True)
    scales.append(float(scale))
    low = int(file.stem) if chunk == 0 else (int(file.stem) + int(files[chunk-1].stem) + 96) // 2
    high = int(file.stem) + len(data["frame_ids"]) if chunk == len(files)-1 else (int(file.stem) + 96 + int(files[chunk+1].stem)) // 2
    for i, frame in enumerate(data["frame_ids"]):
        frame = int(frame)
        if not low <= frame < high or frame not in images:
            continue
        r, t, k, obs = images[frame]
        depth = data["depth"][i] * scale
        frame_depths[frame] = float(np.median(depth))
        obs = obs[obs[:, 2] >= 0]
        # Fixture is 1280x720 -> 518x294. No vertical crop for this input.
        u, v = (obs[:, 0] * 518 / 1280).astype("float32"), (obs[:, 1] * 294 / 720).astype("float32")
        z = cv2.remap(depth, u[:, None], v[:, None], cv2.INTER_LINEAR).ravel()
        confidence = cv2.remap(data["confidence"][i], u[:, None], v[:, None], cv2.INTER_LINEAR).ravel()
        good = (z > 0) & (confidence >= np.quantile(data["confidence"][i], .25))
        rays = np.column_stack([obs[:, :2], np.ones(len(obs))]) @ np.linalg.inv(k).T
        offsets = (rays * z[:, None]) @ r
        for observation, offset in zip(obs[good], offsets[good]):
            tracks[int(observation[2])].append((frame, offset, observation[:2]))
    previous = {k: data[k] for k in ("frame_ids", "depth")}

pair_samples = defaultdict(list)
for track, observations in tracks.items():
    if track % 5 == 0:
        continue  # Independent feature-track subset for evaluation.
    observations.sort(key=lambda x: x[0])
    pairs = list(zip(observations[:-1], observations[1:]))
    if len(observations) > 2:
        pairs.append((observations[0], observations[-1]))
    for a, b in pairs:
        pair_samples[a[0], b[0]].append(a[1] - b[1])
edges = []
for (a, b), samples in pair_samples.items():
    if len(samples) < 12:
        continue
    samples = np.array(samples)
    center = np.median(samples, axis=0)
    residual = np.linalg.norm(samples-center, axis=1)
    good = residual < max(np.median(residual)*3, 1e-5)
    displacement = np.mean(samples[good], axis=0)
    depth = (frame_depths[a] + frame_depths[b])/2
    edges.append((a, b, displacement, min(int(good.sum()), 100) / depth**2, depth))
ids = sorted(images)
index = {frame: i for i, frame in enumerate(ids)}
rows, columns, values = [], [], []
for i, (a,b, *_rest) in enumerate(edges):
    rows.extend([i,i]);columns.extend([index[a], index[b]]);values.extend([-1,1])
rows.append(len(edges));columns.append(0);values.append(1)
matrix = coo_matrix((values,(rows,columns)),shape=(len(edges)+1,len(ids))).tocsr()
rhs = np.vstack([e[2] for e in edges] + [np.zeros(3)])
base_weights = np.array([e[3] for e in edges] + [1e5])
depths = np.array([e[4] for e in edges])
weights = base_weights.copy()
for iteration in range(5):
    weighted = matrix.multiply(np.sqrt(weights)[:,None]).tocsr()
    centers = np.column_stack([lsqr(weighted, rhs[:,j]*np.sqrt(weights),atol=1e-9,btol=1e-9,iter_lim=5000)[0] for j in range(3)])
    error = np.linalg.norm(matrix@centers-rhs, axis=1)
    weights[:-1] = base_weights[:-1] * np.minimum(1,.02*depths/np.maximum(error[:-1],1e-8))
    print("solve",iteration,"edges",len(edges),"relative residual",np.quantile(error[:-1]/depths,[.5,.9,.95]).tolist(),flush=True)

validation = []
for track, observations in tracks.items():
    if track % 5 or len(observations) < 3:
        continue
    # Triangulated using only other observations in the held-out track.
    world = np.array([centers[index[f]] + p for f,p,_ in observations])
    for i,(frame, _offset, observed) in enumerate(observations):
        xyz = np.median(np.delete(world,i,axis=0),axis=0)
        r,t,k,_ = images[frame]
        camera = r@(xyz-centers[index[frame]])
        if camera[2] > 0:
            projected = k@camera;projected=projected[:2]/projected[2]
            validation.append(float(np.linalg.norm(projected-observed)))
print("heldout pixel reprojection",len(validation),np.quantile(validation,[.5,.9,.95]).tolist(),flush=True)
np.savez_compressed(output/"solution.npz",frame_ids=ids,centers=centers,scales=scales)
(output/"report.json").write_text(json.dumps({"edges":len(edges),"frame_ids":ids,"scales":scales,"heldout_track_observations":len(validation),"heldout_pixel_error_quantiles":np.quantile(validation,[.5,.9,.95]).tolist(),"global_bbox":np.ptp(centers,axis=0).tolist()},indent=2)+"\n")
