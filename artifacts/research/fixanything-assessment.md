# FixAnything and faithful property reconstruction

Reviewed on 2026-09-09 in response to the supplied repository.

## Released output

[FixAnything](https://github.com/kvuong2711/fix-anything) refines videos rendered from meshes, Gaussian splats, NeRFs or sparse point clouds.
Released inference produces refined MP4 videos.
Its optional MapAnything stage first produces `reconstruction.glb`; the subsequent FixAnything stage does not export a corrected mesh.
The standard input contains 61 frames resized to 832 by 480, with the first and last frames treated as clean anchors by default.
This makes it relevant to property walkthrough presentation, but a successful video would not itself satisfy the required faithful building mesh.

The [inference script](https://github.com/kvuong2711/fix-anything/blob/main/scripts/run_inference.py) makes this concrete: `fix_video` accepts image frames and returns generated and input image frames.
It does not accept editable mesh vertices or produce replacement geometry.

## Relevant ideas and evidence

The [project page](https://fix-anything.github.io/) describes Wan2.1-I2V-14B adapted with LoRA and masks identifying trusted frames.
A second training stage ranks generated videos by the accuracy of camera poses recovered using COLMAP.
The authors report an AUC@5-degree improvement from 61.1 to 68.3 after this preference optimization.
That evaluates camera consistency, not independently measured wall locations or dimensions.

The project also estimates uncertainty by comparing five random seeds.
Variation correlates with reconstruction error in its reported experiment.
Agreement between generated samples still cannot certify that an unobserved door, wall or object exists.
A plausible, consistent invention can remain wrong about the actual property.

An [open user report](https://github.com/kvuong2711/fix-anything/issues/3) describes invented moving vehicles in an empty-road scene.
This is an unverified report with no author response visible during inspection, not a measured failure rate.
The project itself also documents hallucination examples and the purpose of its anchor masks and preference optimization.

## Proposed integration

Keep observed geometry, source photographs and generated presentation results separately identifiable.
Use the source-camera mesh renderer to prepare bounded trajectories anchored by actual captured images.
Evaluate a refined video against additional captured views excluded from its clean anchors.
Reject changes to visible doors, windows, wall boundaries, furniture and signs even when the output looks sharper.

A geometry-preserving texture experiment is possible: project accepted color refinements only onto already observed mesh surfaces, then test those colors from independent source views.
This is a proposed extension, not an implemented FixAnything capability or a guarantee of faithful textures.
Filling missing rooms or changing mesh positions from generated views would require separate reconstruction and independent validation.
Current camera, depth and window-selection errors must be resolved before appearance refinement can be evaluated meaningfully.

## Mac feasibility and local status

The published setup uses CUDA PyTorch and optionally FlashAttention-2.
The repository estimates approximately 60 GB for the base Wan download, plus the FixAnything adapter.
The loader exposes a device parameter and CPU offloading, but no Mac execution result was established in this review.
A device parameter alone does not verify DiffSynth, attention, offloading hooks or runtime memory behavior on MPS.

The repository identifies its code and FixAnything weights as Apache 2.0.
Code and documentation were inspected; weights were not downloaded and inference was not run locally.
No generated FixAnything frames were inserted into the reconstruction or validation evidence.

## Decision

Retain FixAnything as an optional appearance experiment.
The required output remains an observed, validated building mesh with separately established scale and dimensions.
Local reconstruction findings are recorded in [the implementation report](mac-reconstruction-results.md).
