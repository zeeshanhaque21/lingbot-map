# FixAnything and faithful property reconstruction

Reviewed on 2026-09-09 in response to the supplied repository.

## Released output

[FixAnything](https://github.com/kvuong2711/fix-anything) refines videos rendered from meshes, Gaussian splats, NeRFs or sparse point clouds.
Released inference produces refined MP4 videos.
Its optional MapAnything stage first produces `reconstruction.glb`; the subsequent FixAnything stage does not export a corrected mesh.
The standard input contains 61 frames resized to 832 by 480, with the first and last frames treated as clean anchors by default.
This makes it relevant to property walkthrough presentation.
The user subsequently narrowed the product target to a rough cleaned mesh with loop constraints and fixed 360 stations, so a refined panorama is now a primary output.

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
Camera errors that change room connectivity still need correction, but survey accuracy and photorealistic mesh textures are not prerequisites for this rough-tour PoC.

## Implemented spherical tour experiment

The Mac PoC now exports a 150,000-triangle navigation mesh and six 2048-by-1024 spherical renders from the supplied indoor loop.
Tiny isolated mesh components are removed without filling unobserved rooms.
An audit checks saved return-path matches against the actual exported cameras before adding a navigation edge across the loop.
The candidate has 160 geometrically verified return pairs, with a median pair error of 3.02 pixels compared with 9.98 pixels for the unchanged baseline cameras.
These are training-match consistency values, not independent building measurements.

The new renderer exports 61 perspective images covering both poles and all longitudes at one fixed camera center.
The first and last image use the same captured photograph, resized with matching intrinsics, and are supplied as clean anchors.
Known camera rotations allow refined images to be stitched back into a spherical panorama without estimating replacement geometry.
The raw-render baseline covers every spherical direction, with mean overlap RGB standard deviation 0.00075 and wrap-edge RGB difference 0.00195 on a 0-to-1 scale.
These small numerical values validate the rendering and stitching path; they do not predict FixAnything's generated-image consistency.
Only 36% to 41% of panorama pixels currently hit observed surfaces, because the forward-facing capture does not see much of the ceiling and floor.
Generated fill must remain distinguishable from observed building evidence.

## Mac feasibility and local status

The published setup uses CUDA PyTorch and optionally FlashAttention-2.
The repository estimates approximately 60 GB for the base Wan download, plus the FixAnything adapter.
The actual selected Hugging Face files total 82,856,630,348 bytes, including the base shards, encoders, tokenizer, VAE and adapter.
Downloads are in progress through Motrix with published SHA-256 checksums where available.
The loader exposes a device parameter and CPU offloading, but a completed Mac execution result has not yet been established.
A device parameter alone does not verify DiffSynth, attention, offloading hooks or runtime memory behavior on MPS.

The repository identifies its code and FixAnything weights as Apache 2.0.
Code and documentation were inspected, and isolated local model setup is in progress.
No generated FixAnything frames were inserted into the reconstruction or validation evidence.

## Decision

Use FixAnything as the candidate appearance stage for fixed spherical tour stations.
The mesh provides rough spatial navigation and room connectivity; the panoramas provide the viewing experience.
Do not claim accurate dimensions or faithful unobserved details from this PoC.
Local reconstruction findings are recorded in [the implementation report](mac-reconstruction-results.md).
