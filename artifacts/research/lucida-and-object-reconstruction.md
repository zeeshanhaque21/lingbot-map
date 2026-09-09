# Lucida and object reconstruction for a faithful property model

Research checked on 2026-09-09.

Lucida is relevant to editable furniture and object assembly.
The building shell still requires reliable camera poses, depth, and cross-room registration.
The proposed integration is to preserve an observed building mesh, collect per-object evidence from the same coordinates, and validate any completed object against all available views before assembly.

## What Lucida actually contributes

Lucida starts from posed RGB or RGB-D observations, builds per-object evidence bundles, generates complete assets, and refines their placement with GizmoAct.
Its generator synthesizes an isolated object image from multiple observations and uses Seed3D 2.0 to obtain the asset.
GizmoAct is a trained visual policy with supervised and reinforcement learning stages, rather than an ordinary prompt that can be substituted without evaluation.
It refines rotation, translation, and anisotropic scale through rendered feedback.
[Paper](https://arxiv.org/html/2608.30821v1), [project](https://lucida-r2s.github.io/).

The reported scene F-score of 0.924 uses a 0.1 m distance threshold.
Object scores normalize and align individual objects, so they do not directly establish building dimensions.
The scene experiment also gives Lucida multi-view evidence while the listed SceneGen and SAM 3D baselines receive a reference image and instance masks.
The improvement is useful evidence for the complete system, with that input difference included.
[Evaluation protocol](https://arxiv.org/html/2608.30821v1).

The project page inspected here links the paper and demonstrations; it did not expose a runnable implementation or policy checkpoint link in the visible page.
This is a release-status observation, not proof that no release exists elsewhere.
[Project page](https://lucida-r2s.github.io/).

## Candidate components

| Component | Inputs and output | Fit for this app | Mac and adoption constraint |
|---|---|---|---|
| ShapeR | Sparse object points, posed views and captions to object mesh | Closest match to the requested point-cloud-plus-pictures direction | Official setup requires CUDA, flash attention and TorchSparse; repository states predominantly CC-BY-NC |
| SAM 3D Objects | Masked image to complete object geometry, appearance and pose | Object proposals and completion | SAM License; complete shape includes inferred hidden geometry |
| TRELLIS.2 | Image to textured object asset; also shape-conditioned appearance | Detailed furniture assets and optional texturing research | Official implementation uses Linux/CUDA and several custom GPU packages; repository states model and code are MIT |
| NKSR | Oriented point cloud to implicit surface and mesh | Learned surface reconstruction for scanned objects or scenes | Official implementation includes CUDA extensions; code and checkpoint terms differ |
| TSDF fusion | Posed depth and color to observed surface mesh | Implemented Mac baseline; keeps observation provenance | CPU fusion on this Mac; missing observations remain holes |

Sources: [ShapeR repository](https://github.com/facebookresearch/ShapeR), [ShapeR installation](https://github.com/facebookresearch/ShapeR/blob/main/INSTALL.md), [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), [TRELLIS.2 repository](https://github.com/microsoft/TRELLIS.2), [NKSR repository](https://github.com/nv-tlabs/NKSR), [Open3D integration](https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html).

ShapeR explicitly conditions on metric sparse points and camera observations.
That metric information comes from capture preprocessing; a monocular video with unknown scale does not gain independently verified dimensions merely by passing through ShapeR.
Its published evaluation includes 178 objects in seven real scenes.
[ShapeR paper](https://arxiv.org/html/2601.11514v1).

TRELLIS.2's project page describes its page materials as research-only, while the repository explicitly labels model and code MIT.
The actual asset and dependency licenses should be checked individually when selecting a product component.
The project-page statement alone should not be used to classify the code as noncommercial.
[Project](https://microsoft.github.io/TRELLIS.2/), [repository](https://github.com/microsoft/TRELLIS.2).

## Original hybrid design

The following is a proposed combination, not a reproduced Lucida result.

1. Reconstruct the observed scene with camera, depth, confidence and source-frame records.
2. Recover stable global poses using visual feature tracks and bundle adjustment, and compare with the learned trajectory.
3. Store each object's evidence as source views, masks, partial points, an initial box and its placement in the building coordinate system.
4. Prefer direct surface reconstruction when the object is sufficiently observed.
5. When completion is necessary, treat the generated mesh as a hypothesis with observed and inferred regions separately labelled.
6. Refine placement using multiple views, surface residuals, silhouette agreement and free-space violations.
7. Keep the unchanged observed mesh as the do-nothing baseline and reject replacements that worsen agreement on reserved views.

Anisotropic scaling can make an approximate object fit better while changing its proportions.
For a property record, every such transform must be retained and any dimension claim must be tested against independent measurements.
Wall boundaries, openings and room connectivity must not inherit dimensions from a generated sofa or a completed room image.

## Local implementation evidence so far

The Mac now runs the official long Lingbot checkpoint through PyTorch MPS and exports an actual triangle mesh, point cloud, GLB, camera records and source-view render comparisons.
The checkpoint SHA-256 is `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`.
The available test video is the 500-second furniture-store walkthrough `indoor_travel.MP4`, not a supplied residential property scan.

A 12-frame float32 run completed inference in 5.98 seconds, excluding model loading and frame preparation.
A paired bfloat16 run showed 0.246% median and 0.677% 95th-percentile raw relative depth difference from float32 across those frames.
These are single-run numerical comparisons with a float32 baseline, not accuracy against measured building geometry.

The first float32 full-video attempt was terminated by signal 9 at frame 44 during rising memory pressure.
Bfloat16 processing is being evaluated for the complete sequence.
The early mesh had substantial holes, and its pose convention was subsequently found to be incorrect for the tested checkpoint.
Those early coverage figures must not be treated as a fair learned-pose baseline.
Current run status and eventual final quality findings belong in the implementation report and generated validation artifacts.

## Checkpoint convention discovered during implementation

The official long checkpoint hash above produces decoded camera-to-world matrices, although the upstream pose-decoding helper describes world-to-camera output.
Using the wrong interpretation caused severely inconsistent fusion while all tensor shapes and ordinary geometry unit tests still passed.
On 80 frames matched to a separate COLMAP reconstruction, a single similarity alignment gave median orientation disagreement of 106.07 degrees under the documented interpretation and 2.33 degrees under the camera-to-world interpretation.
Median camera-center residual, divided by the reference trajectory's bounding-box diagonal, changed from 19.31% to 0.769%.
These are comparisons against an estimated photogrammetry model, not surveyed ground truth.

This is a local finding for the exact tested checkpoint, not a universal statement about all Lingbot releases.
The adapter now identifies that checkpoint by hash and normalizes its pose matrices at the inference boundary.
Existing experimental archives are preserved and corrected into separate directories.
