# Mac property reconstruction

This pipeline converts a walkthrough video into an observed-surface triangle mesh, a GLB asset, a point cloud, and camera-linked validation evidence.
It runs inference on Apple Silicon through PyTorch MPS and fuses depth on CPU through Open3D.
It does not certify dimensions from an uncalibrated monocular video.

## Setup

Use Python 3.12 and install the reconstruction dependencies in a dedicated environment.

```sh
uv venv .venv-reconstruction --python 3.12
uv pip install --python .venv-reconstruction/bin/python -r requirements-reconstruction.txt
brew install ffmpeg colmap
```

Place the official `lingbot-map-long.pt` checkpoint in `checkpoints/`.
The tested file has SHA-256 `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`.
The model source is [Robbyant's model repository](https://huggingface.co/robbyant/lingbot-map).

## Run and inspect

```sh
.venv-reconstruction/bin/python -m lingbot_map.reconstruction run \
  --video /path/to/walkthrough.mp4 --output reconstructions/property

.venv-reconstruction/bin/python -m lingbot_map.reconstruction view \
  --output reconstructions/property/final
```

The viewer opens at `http://127.0.0.1:8081`.
Select a captured frame and use **Compare from this camera** to inspect the reconstructed surface from its source viewpoint.
The GLB download uses the actual mesh, not a point-cloud placeholder.

The default sampling rate is 2 fps with 96-frame inference windows and 24 shared frames.
The model retains its 8 scale frames, 64-frame cache and 518-pixel inference width.
Bfloat16 is the Mac default; camera and depth heads retain float32 arithmetic.
Metal allocation telemetry is appended to `memory.jsonl`.

For individual stages:

```sh
.venv-reconstruction/bin/python -m lingbot_map.reconstruction prepare --video /path/to/video.mp4 --output reconstructions/property
.venv-reconstruction/bin/python -m lingbot_map.reconstruction infer --output reconstructions/property
.venv-reconstruction/bin/python -m lingbot_map.reconstruction fuse --output reconstructions/property
.venv-reconstruction/bin/python -m lingbot_map.reconstruction validate --output reconstructions/property
.venv-reconstruction/bin/python -m lingbot_map.reconstruction sfm --output reconstructions/property
```

`run` reconstructs global COLMAP cameras, calibrates learned depth against triangulated points, and writes the final artifacts under `final/model/`.
It refuses to assemble disconnected camera models into an apparently complete building.
`sfm` builds feature tracks and global bundle-adjusted cameras with COLMAP 4 on CPU.
Use `--mapper incremental` to reproduce the incremental comparison.
Photogrammetry results remain in `colmap/`, including disconnected models and intermediate snapshots.

Existing inference can be refined explicitly with a selected photogrammetry model.

```sh
.venv-reconstruction/bin/python -m lingbot_map.reconstruction refine \
  --source reconstructions/property --output reconstructions/property-refined \
  --colmap-model reconstructions/property/colmap/global/0/text
.venv-reconstruction/bin/python -m lingbot_map.reconstruction fuse --output reconstructions/property-refined
.venv-reconstruction/bin/python -m lingbot_map.reconstruction validate --output reconstructions/property-refined
```

## Artifacts and evidence

| Artifact | Meaning |
|---|---|
| `input.json` | Source hash, probe metadata, sampling configuration and timestamps |
| `windows/*.npz` | RGB, depth, confidence, intrinsics, extrinsics and frame IDs |
| `model/property.glb` | Simplified colored triangle mesh for app ingestion |
| `model/observed-surfaces.ply` | Full-resolution fused surface mesh |
| `model/observed-points.ply` | Fused observed points |
| `model/cameras.json` | Camera poses, processed-image intrinsics and source timestamps |
| `model/alignment.json` | Window transforms and registration residuals |
| `model/frame-validation.json` | Accepted depth support and per-frame diagnostics |
| `model/validation.json` | Coverage, hashes, geometry counts and explicit limitations |
| `model/render-validation.json` | Source-view mesh coverage and appearance checks |
| `model/source-comparison-*.jpg` | Source images beside rendered geometry and difference maps |

Input extraction and completed inference windows are resumable.
A changed input or inference configuration requires a separate output directory.
Inference currently saves at each completed window; an interrupted current window is recomputed.
Completed windows are written through temporary files followed by atomic replacement.

## Fidelity policy

The pipeline rejects uncertain depths, depth discontinuities and samples lacking agreement with another camera view.
The production path uses a common photogrammetry coordinate system for all windows.
The experimental learned-pose path instead aligns shared-frame geometry with a robust similarity transform and refuses joins whose residuals exceed its gate.
TSDF fusion creates an open surface mesh without watertight completion or invented room boundaries.
Every tenth frame, with index ending in 5, is withheld from fusion for render comparison.
Those frames still participate in pose/depth inference, so the check is not independent ground truth.

Stored extrinsics are explicitly world-to-camera matrices in the OpenCV axis convention.
The tested long checkpoint actually decodes camera-to-world poses despite the upstream helper's documentation.
The adapter normalizes that convention before saving any depth geometry.
This was checked against 80 independently estimated photogrammetry cameras and multi-view depth consistency.
Unknown checkpoint hashes require an explicit `--pose-convention` instead of inheriting an unverified assumption.
GLB vertices receive the explicit transform `diag(1,-1,-1)` for viewing in a Y-up application.
That transform does not establish gravity or metric scale.
All distances remain in model units until an external measured reference is supplied and validated.

Sequential learned-window registration does not perform global loop closure.
The photogrammetry path jointly optimizes the connected visual track graph; its current matcher uses temporal and quadratic pairs without a dedicated place-recognition loop detector.
Reflective floors, glass, people, textureless walls and unseen ceilings can leave missing or incorrect geometry.
The generated report keeps `ready_for_verified_property_listing` false until independent dimensions, room connectivity and missing-surface checks are supplied.

Object-completion options, Lucida's role, and the limits of its reported metrics are documented in [the research follow-up](artifacts/research/lucida-and-object-reconstruction.md).
