# Mac property reconstruction

This pipeline converts a walkthrough video into an observed-surface triangle mesh, a GLB asset, a point cloud, and camera-linked validation evidence.
It runs inference on Apple Silicon through PyTorch MPS and fuses depth on CPU through Open3D.
It does not certify dimensions from an uncalibrated monocular video.

Current status is a research prototype.
The current real-estate PoC target is a cleaned rough mesh with loop constraints and fixed 360 viewing stations.
The [property tour report](artifacts/research/property-tour-poc.md) documents the working local tour and pending FixAnything refinement.
The short integration capture passes internal screening, but every tested full-walkthrough variant still fails fidelity checks.
The [implementation report](artifacts/research/mac-reconstruction-results.md) identifies the current review asset and its remaining failures.
The [Brush experiment report](artifacts/research/photometric-and-brush-results.md) includes native Metal Gaussian training, calibrated dataset export and reserved-image comparisons.
Its current result covers a 24-second section and does not replace the unresolved full-building geometry checks.
The [Gaussian surface report](artifacts/research/gaussian-surface-results.md) covers depth extraction, pixel conventions and failed mesh variants.
The [OpenMVS report](artifacts/research/openmvs-mac-results.md) provides a tested native CPU path from calibrated photographs to a complete textured GLB, with reserved-view evidence and remaining geometry failures.

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
  --output reconstructions/property
```

The viewer opens at `http://127.0.0.1:8081`.
Select a captured frame and use **Compare from this camera** to inspect the reconstructed surface from its source viewpoint.
The GLB download uses the actual mesh, not a point-cloud placeholder.
The viewer displays failed geometry checks and provides buttons for the weakest inspected viewpoints.

An existing ordered image sequence can be supplied directly.
PNG and JPEG files are copied byte-for-byte in natural filename order, with source hashes and no resizing or recompression.
Select one sequence directory, such as `example/loop`, rather than the parent containing several captures.
Image sequences retain unknown capture times and use frame numbers in the viewer.
Adaptive extraction of extra video frames is unavailable for this input type.

```sh
.venv-reconstruction/bin/python -m lingbot_map.reconstruction run \
  --images example/loop --output reconstructions/examples/loop \
  --camera-source learned --motion-weight 10 --mapper global
```

The default sampling rate is 2 fps with 96-frame inference windows and 24 shared frames.
The model retains its 8 scale frames, 64-frame cache and 518-pixel inference width.
Bfloat16 is the Mac default; camera and depth heads retain float32 arithmetic.
Metal allocation telemetry is appended to `memory.jsonl`.

For individual stages:

```sh
.venv-reconstruction/bin/python -m lingbot_map.reconstruction prepare --video /path/to/video.mp4 --output reconstructions/property
.venv-reconstruction/bin/python -m lingbot_map.reconstruction infer --output reconstructions/property
.venv-reconstruction/bin/python -m lingbot_map.reconstruction sfm --output reconstructions/property
.venv-reconstruction/bin/python -m lingbot_map.reconstruction register --source reconstructions/property --colmap-model reconstructions/property/colmap/global/0/text --output reconstructions/property/final
.venv-reconstruction/bin/python -m lingbot_map.reconstruction fuse --output reconstructions/property/final
.venv-reconstruction/bin/python -m lingbot_map.reconstruction validate --output reconstructions/property/final
```

`run` builds COLMAP tracks and global rotations, constrains camera translations using learned depths in a shared scale, and writes the final artifacts under `final/model/`.
It refuses to assemble disconnected camera models into an apparently complete building.
Depth scales are aligned through shared images between inference windows.
A robust translation graph then fits multi-view feature correspondences, reserving one fifth of feature tracks for evaluation.
Reliable photogrammetry rotation groups are aligned to overlapping learned orientations.
When the graph is disconnected, `run` extracts 10 fps bursts around short gaps, verifies their continuous depth agreement, and retries registration.
Every unsuccessful attempt is preserved separately.
`pipeline-result.json` identifies the successful output directory, and the viewer accepts the run's root directory.
Adaptive repair currently requires a 2 fps base capture and permits two repair rounds.
This hybrid is necessary because unconstrained monocular photogrammetry can collapse scale over a long walkthrough while retaining small reprojection errors.
`sfm` builds feature tracks and global bundle-adjusted cameras with COLMAP 4 on CPU.
Use `--mapper incremental` to reproduce the incremental comparison.
Photogrammetry results remain in `colmap/`, including disconnected models and intermediate snapshots.

Existing inference can be registered explicitly with a selected photogrammetry model.

```sh
.venv-reconstruction/bin/python -m lingbot_map.reconstruction register \
  --source reconstructions/property --output reconstructions/property-refined \
  --colmap-model reconstructions/property/colmap/global/0/text
.venv-reconstruction/bin/python -m lingbot_map.reconstruction fuse --output reconstructions/property-refined
.venv-reconstruction/bin/python -m lingbot_map.reconstruction validate --output reconstructions/property-refined
```

Use `repair` in place of `register` to enable adaptive sampling for an existing inference run.
It also accepts `--bridges /path/to/bridges.json` for previously validated dense bursts.
Registration diagnostics retain the tested graph edges, rejected motions, bridge checks and withheld feature-track errors.

`--camera-source learned` is an experimental registration alternative for `run`, `register` and `repair`.
It retains the learned intrinsics associated with each depth map and uses aligned learned orientations with photogrammetry feature tracks.
It also selects newer overlapping predictions after the checkpoint's eight-frame scale bootstrap, using the same ownership for depth, intrinsics, orientations, feature tracks and fusion.
`--motion-weight 10` increases the influence of depth-validated local motion relative to photogrammetry feature edges for a controlled comparison.
Use a new output directory to compare it with the default `sfm` camera source.
The local camera falsifier in the implementation report motivates this option; its complete-scene validation remains separate from the default baseline.

For an uninterrupted native-checkpoint comparison, `run --camera-source native --window 1000` skips external photogrammetry and registration.
Choose a window that covers the extracted sequence when testing an uninterrupted stream.
Long windows retain more input images and CPU predictions and only save when the window completes.
This comparison is experimental and must pass the same exported-mesh checks.
On the available 1,000-frame store walkthrough, the uninterrupted native export failed 92 of 100 reserved viewpoints.
The hybrid with `--camera-source learned --motion-weight 10` reduced that count to seven, but also failed the fidelity gate.

## Fixed 360 tours

An existing colored surface reconstruction can be exported as a lightweight navigation mesh and spherical stations.

```sh
.venv-reconstruction/bin/python -m lingbot_map.reconstruction tour \
  --source reconstructions/examples/loop-closure-v1/final \
  --output reconstructions/tours/loop-new --stations 6
.venv-reconstruction/bin/python -m lingbot_map.reconstruction tour-view \
  --output reconstructions/tours/loop-new --port 8083
```

The viewer serves the tour at `http://127.0.0.1:8083/` without external web dependencies.
Click a mesh marker or station thumbnail to enter its panorama.
Drag to look around, scroll to zoom, and use **3D overview** to return to the model.
Each station preserves its original image, pose, raw panorama, radial depth and observed-surface mask.
Cleanup removes tiny isolated components and simplifies the navigation mesh while retaining a detailed rendering mesh.
A return-path edge requires an audit of successful matching, registered endpoint cameras and consistent feature matches.
For arbitrary captures without this evidence, the exported tour remains an open path.

`tour-sweep --source <tour> --station 000 --output <new-sweep>` exports 61 calibrated camera orientations covering the full sphere.
The first and last frame use the same captured station photograph as FixAnything clean anchors.
For a horizontal pass, use `--sweep-mode horizontal --sweep-up X Y Z` with the scene-up direction in reconstruction coordinates.
The camera turns around that axis while retaining its captured tilt and fixed position.
`--sweep-frames` sets the horizontal frame count, including the repeated endpoint; 61 frames give six-degree steps.
The horizontal ring does not cover the full sphere or its poles.
For the supplied office loop, the camera-trajectory plane gives the estimated up direction used in this tested command:

```sh
.venv-reconstruction/bin/python -m lingbot_map.reconstruction tour-sweep \
  --source reconstructions/tours/loop-v1 --station 000 \
  --output reconstructions/tours/loop-v1/sweep-000-horizontal-v1 \
  --sweep-mode horizontal --sweep-frames 61 \
  --sweep-up -0.000990950758682904 -0.940829700957784 -0.3388782846573102
```

This up direction is an estimate for this capture, not a universal axis or measured gravity.
The completed horizontal pass contains 61 images at 832 by 480, with zero entirely empty renders and minimum observed coverage of 18.37%.
`tour-stitch --source <sweep> --images <frame-directory> --output <new-directory>` reprojects those images with their saved rotations.
Add `--generated` when stitching model-generated frames.
The raw-render stitch is the do-nothing baseline for overlap and wrap-seam comparisons.
Before loading FixAnything weights, inspect the proposed input with:

```sh
.venv-fixanything/bin/python artifacts/research/experiments/run_fixanything_mac.py \
  --sweep <sweep> --check-input
```

This rejects incomplete spherical clips and renders with no observed surfaces.
The original spherical station-000 sweep fails this check; the horizontal pass passes the input check but still requires output generation and visual validation.
The previous generation was stopped; no completed generated panorama is available.
Validate one complete generated, stitched and visually inspected 360 station against source photos before processing more stations or increasing run cost.
A partial runtime smoke test does not meet this gate.
The [conditioning investigation](artifacts/research/fixanything-panorama-debugging.md) records the evidence and next corrections.
Model downloads and the isolated MPS adapter are described in the [FixAnything assessment](artifacts/research/fixanything-assessment.md).

## Reconstruction artifacts

| Artifact | Meaning |
|---|---|
| `input.json` | Source hash, probe metadata, sampling configuration and timestamps |
| `windows/*.npz` | RGB, depth, confidence, intrinsics, extrinsics and frame IDs |
| `registration.json` | Depth overlap checks, connected cameras and withheld feature-track errors |
| `pipeline-result.json` | Actual completed registration directory after any adaptive repairs |
| `model/property.glb` | Simplified colored triangle mesh for app ingestion |
| `model/property-detail.glb` | Optional detailed re-export of an existing reconstruction |
| `model/observed-surfaces.ply` | Full-resolution fused surface mesh |
| `model/observed-points.ply` | Fused observed points |
| `model/cameras.json` | Camera poses, processed-image intrinsics and source timestamps |
| `model/alignment.json` | Window transforms and registration residuals |
| `model/frame-validation.json` | Accepted depth support and per-frame diagnostics |
| `model/validation.json` | Coverage, hashes, geometry counts and explicit limitations |
| `model/render-validation.json` | Source-view mesh coverage and appearance checks |
| `model/web-render-validation.json` | The same checks on the actual exported GLB |
| `model/source-comparison-*.jpg` | Source images beside rendered geometry and difference maps |

Input extraction and completed inference windows are resumable.
A changed input or inference configuration requires a separate output directory.
Inference currently saves at each completed window; an interrupted current window is recomputed.
Completed windows are written through temporary files followed by atomic replacement.
`export --output <run>` produces a detailed app mesh from saved surfaces without repeating inference or camera reconstruction.
The triangle budget grows with capture length, and `app-export.json` records the actual size and provenance.

## Fidelity policy

The pipeline rejects uncertain depths, depth discontinuities and samples lacking agreement with another camera view.
The default path uses global photogrammetry rotations and a depth-constrained translation graph for all windows.
The experimental `refine` stage instead scales each learned depth map to raw SfM points; it remains available for controlled comparisons and can inherit SfM scale collapse.
The experimental learned-pose path instead aligns shared-frame geometry with a robust similarity transform and refuses joins whose residuals exceed its gate.
TSDF fusion creates an open surface mesh without watertight completion or invented room boundaries.
Every tenth frame, with index ending in 5, is withheld from fusion for render comparison.
Those frames still participate in pose/depth inference, so the check is not independent ground truth.
Validation checks rendered depth agreement and color error as well as coverage; a large incorrect plane cannot pass just by covering the image.
It checks the least-supported sampled view as well as median results.
Validation now inspects every reserved frame by default, since the earlier 24-view sample missed additional local failures.
The CLI returns exit status 2 with `status: needs_review` when these checks fail.
Its numeric screening thresholds are engineering defaults, not calibrated guarantees of building accuracy.

Stored extrinsics are explicitly world-to-camera matrices in the OpenCV axis convention.
The tested long checkpoint actually decodes camera-to-world poses despite the upstream helper's documentation.
The adapter normalizes that convention before saving any depth geometry.
This was checked against 80 independently estimated photogrammetry cameras and multi-view depth consistency.
Unknown checkpoint hashes require an explicit `--pose-convention` instead of inheriting an unverified assumption.
GLB vertices receive the explicit transform `diag(1,-1,-1)` for viewing in a Y-up application.
That transform does not establish gravity or metric scale.
The GLB uses `KHR_materials_unlit` because its colors already contain captured illumination.
The viewer serves the exact GLB bytes, preserving its colors and material metadata.
All distances remain in model units until an external measured reference is supplied and validated.

Sequential learned-window registration does not perform global loop closure.
The photogrammetry path jointly optimizes the connected visual track graph; its current matcher uses temporal and quadratic pairs without a dedicated place-recognition loop detector.
Reflective floors, glass, people, textureless walls and unseen ceilings can leave missing or incorrect geometry.
The generated report keeps `ready_for_verified_property_listing` false until independent dimensions, room connectivity and missing-surface checks are supplied.

Object-completion options, Lucida's role, and the limits of its reported metrics are documented in [the research follow-up](artifacts/research/lucida-and-object-reconstruction.md).
Measured local outcomes and failed alternatives are recorded in [the implementation report](artifacts/research/mac-reconstruction-results.md).
FixAnything's released video output, proposed role and Mac limitations are assessed in [a separate review](artifacts/research/fixanything-assessment.md).
