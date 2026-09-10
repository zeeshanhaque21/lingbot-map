# Indoor reconstruction from omnidirectional video

Researched on 2026-09-09 for the M3 Max Mac and the real estate prototype.

## Recommendation

The best next experiment is **COLMAP's calibrated panorama workflow, followed by OpenMVS surface reconstruction, with original captured panoramas as tour stations**.
This is the strongest practical fit found for our existing Mac pipeline, not a demonstrated quality winner on our footage.
The existing LingBot pilot remains the measured baseline; none of the alternatives below has yet produced a validated replacement mesh here.

For longer walkthroughs, add explicit matching of revisited places and global bundle adjustment.
Use **stella_vslam** as a native spherical tracking and loop-closure alternative if offline camera recovery fails or long sequences need a tracking frontend.
Do not run two independent camera estimators and combine their outputs without an explicit alignment and consistency check.

Keep the product's outputs distinct: a source-supported navigation mesh, captured 360 stations, and optional appearance improvement.
An attractive generated panorama does not establish correct wall positions or room connections.

## Why this changes our current approach

The [AIST pilot](aist-indoor-omni-pilot.md) used 24 timestamps and four directions per timestamp.
LingBot estimated separate camera poses and intrinsics for those 96 images instead of respecting the known panoramic camera.
The audit found a maximum same-time relative rotation error of 10.32 degrees and median focal disagreement of 8.66%.
Both meshes failed fidelity screening, despite improved spherical coverage from the extra directions.

A panorama contains many viewing directions from approximately one optical center.
Depth comes from camera translation between timestamps; rotating or cropping one panorama does not create a stereo baseline.
Consequently, every virtual view from one timestamp must retain its known rotation, calibration, and shared center.
The reconstruction should optimize one camera position per timestamp while using observations across the sphere.
This addresses a measured constraint violation, although it does not prove that camera errors explain every hole.

## What the primary sources establish

| Candidate | Relevant capability | Fit for this project |
| --- | --- | --- |
| [COLMAP panorama workflow](https://colmap.github.io/rigs.html#reconstruction-from-360-spherical-images) | Projects panoramas into virtual cameras with fixed calibration and rig geometry. | First Mac experiment; existing COLMAP installation and OpenMVS integration reduce new moving parts. |
| [stella_vslam](https://stella-cv.readthedocs.io/en/latest/overview.html) | Native equirectangular sparse SLAM; tracking, saved maps, and relocalization. | Strong alternative for camera trajectory and loop closure; still requires a dense surface backend. |
| [OpenMVS](https://github.com/cdcseacave/openMVS) | Takes calibrated cameras and sparse points; reconstructs dense points, mesh, refinement, and texture. | First dense backend; a native arm64 CPU build already exists locally. |
| [360Recon](https://github.com/LeonATP/360Recon) | Panoramic multi-view depth prediction and mesh fusion. | Particularly relevant geometry research; released code restricts commercial use and requires a compiled spherical operator. |
| [PanSplat](https://github.com/chengzhag/PanSplat) | Two panoramas produce Gaussians for novel views; includes real video preprocessing. | Appearance candidate after camera recovery; documented 4K demo requires at least 24 GB GPU memory and CUDA dependencies. |
| [OmniSplat](https://github.com/esw0116/OmniSplat) / [Splatter-360](https://github.com/thucz/splatter360) | Feed-forward Gaussian reconstruction and novel-view rendering from panoramic images. | Research comparators for appearance; neither is a verified loop-closed Mac mesh pipeline here. |
| [Metashape Professional](https://www.agisoft.com/features/professional-edition/) | Spherical camera and multi-camera support within a photogrammetry product. | Useful commercial reference run on identical inputs, if available; no purchase or installation is needed to choose the open-source experiment. |
| [Brush](https://github.com/ArthurBrussee/brush) | Gaussian splatting with macOS support through WebGPU-compatible technology. | Optional appearance branch using validated cameras; does not by itself establish an accurate building surface. |

### COLMAP now has the exact panoramic starting point

The upstream [4.2.0 example](https://github.com/colmap/colmap/blob/4.2.0/python/examples/panorama_sfm.py) exposes a CPU option.
Its [panorama module](https://github.com/colmap/colmap/blob/4.2.0/python/pycolmap/panorama.py) fixes virtual sensor transforms and camera calibration, skips same-frame image pairs, and can convert the sparse result back to one equirectangular camera per capture.
It also includes a direct spherical mode.
These are source inspections, not execution results on this Mac.

The overlapping projection preset includes multiple pitch bands.
Its generated feature masks partition overlapping views; they do not automatically encode AIST's invalid source rectangles.
Our integration must intersect them with the projected source-validity mask.
It must also reconcile COLMAP's half-pixel convention with the integer-centered convention in our existing projector.

The helper's sequential matching option is not sufficient evidence of loop closure.
Add verified long-range matches for revisited rooms, then inspect the optimized trajectory and reconstructed surfaces.
Use a new output directory: the inspected helper deletes an existing database at its output path.

### stella_vslam is unusually relevant to our exact input

The [official tutorial](https://stella-cv.readthedocs.io/en/latest/simple_tutorial.html#tracking-and-mapping) uses AIST Living Lab 1 for mapping and Lab 2 for localization.
Its published example log includes detected loop closure and loop bundle adjustment.
That establishes a documented reference workflow for the files already downloaded; it does not mean our run has reproduced it.

PanSplat's expanded “Use your own video” instructions also use stella_vslam.
They first map the video and then localize against the saved map to export poses.
Thus, even this learned panoramic renderer relies on a separate camera recovery step for real footage.
[Source: PanSplat's official workflow](https://github.com/chengzhag/PanSplat#-data-preparation).

The [dense stella_vslam extension](https://github.com/RoblabWh/stella_vslam_dense) directly combines spherical SLAM with panoramic PatchMatch.
Its documented execution path requires NVIDIA Docker and an NVIDIA GPU, so it is not an established native Mac shortcut.
Its algorithm is relevant; porting that implementation is a separate project.

### 360Recon is the most directly relevant learned mesh candidate found

[360Recon's paper](https://arxiv.org/abs/2411.19102) describes spherical feature extraction, multi-view cost volumes, and depth-based scene reconstruction.
The released repository includes pretrained weights and a mesh-fusion evaluation command.
It acknowledges [SphereSfM](https://github.com/json87/SphereSfM) for real-world panoramic pose recovery.
SphereSfM itself supports spherical sparse reconstruction and conversion to perspective views for dense reconstruction.

However, [360Recon's actual license](https://github.com/LeonATP/360Recon/blob/main/LICENSE) limits the released implementation to non-commercial research and educational use.
That is a concrete mismatch with adopting it directly into the real estate app.
Its documented setup also requires compiling a spherical convolution operator; native MPS support is unverified.
Treat its approach as a research reference, not an immediately deployable replacement.

PanSplat and OmniSplat show MIT repository licenses, while their dependencies and pretrained weights remain separate artifacts to check before adoption.
OpenMVS identifies its license as AGPL-3.0.
These are upstream license labels, not a completed product licensing assessment.

## Proposed pipeline

This combination is an engineering proposal assembled from the verified components above.
Its quality must be established on the small sample before expansion.

1. **Preserve capture evidence.**
   Retain original panoramas, timestamps, frame identities, and invalid-region masks.
   Select sharp frames with actual translation and overlapping scene content.
   Use adaptive keyframes for longer video instead of assuming a larger frame count improves geometry.

2. **Recover calibrated cameras.**
   Begin with the same 24 AIST timestamps and the existing four directions so the first comparison isolates camera recovery.
   Configure a single virtual rig with fixed rotations, zero relative translations, and known intrinsics.
   Match observations between timestamps and jointly optimize camera positions and sparse points.
   For a later coverage experiment, add upward and downward views only where source pixels are valid.

3. **Close real loops before final dense fusion.**
   Match revisits across the trajectory and verify their geometry before accepting them.
   Optimize the connected map globally, then generate or regenerate dense geometry using the final cameras.
   A 12-second fragment with no demonstrated revisit cannot certify loop closure.
   stella_vslam is the next camera alternative if the fixed-rig COLMAP experiment cannot recover a coherent trajectory.

4. **Recompute multi-view geometry.**
   Feed calibrated perspective cameras, source images, and sparse points into OpenMVS.
   Select stereo neighbors with overlap and translation, including useful revisits.
   Reject invalid, occluded, or inconsistent depth observations before surface reconstruction.
   Do not replace poses in LingBot's saved arrays and assume the old depths remain geometrically compatible.
   A spherical depth backend requires explicit conversion between radial range and perspective optical-axis depth.

5. **Construct a conservative navigation mesh.**
   Reconstruct and texture surfaces from source-supported observations.
   Remove isolated fragments and simplify only after validating the full-resolution result.
   A proposed fallback for poorly textured walls is bounded plane fitting using observed points, camera visibility, and multiple source images.
   Preserve observed door and window openings; do not use unrestricted hole filling to fabricate room connectivity.
   Record any inferred patches separately from measured support.

6. **Place stations at captured camera positions.**
   Use actual video panoramas at selected trajectory points.
   Preserve full horizontal wrap and explicit source-invalid regions.
   Reuse sharp observations from nearby times only with depth-aware reprojection and occlusion checks.
   FixAnything can be evaluated later for a supported repair; keep original observed pixels as the appearance reference.

7. **Validate the exported deliverable.**
   Inspect the actual GLB in the viewer and render one complete sphere at a source station.
   Compare against the captured panorama and held-out source observations.
   Keep geometry coverage, correspondence error, source support, and generated-pixel fraction as separate measurements.
   Any optional Gaussian model must use the same accepted camera solution and receive its own appearance evaluation.

## Small-sample acceptance experiment

The first experiment reuses the existing 24 captures from 45 to 57 seconds of Lab 1.
It produces one complete textured GLB plus one source-anchored 360 station, with no additional video downloads.
The second downloaded video remains unused until this deliverable passes.

Compare the unchanged LingBot output with the calibrated COLMAP/OpenMVS candidate on identical source times and masks.
Retain the captured panorama as the do-nothing appearance baseline.
Report the number of registered captures, connected components, feature support, camera residuals, full mesh coverage, simplified mesh coverage, and source-image errors.
Audit known virtual-camera rotations and centers directly.
They should agree within numerical tolerance because they are constrained, but this alone is not evidence of correct scene geometry.

For validation beyond the previous pilot, reserve complete capture times from dense reconstruction, including all their virtual directions.
Estimate their poses against the accepted map without inserting their pixels into dense fusion or texture selection.
Do not call these independent survey measurements; camera localization still uses their images.
Visually check wall alignment, corners, doorways, duplicated surfaces, and the seam at the checked station.
An attractive photo station cannot compensate for a failed mesh.

Only after this complete sample succeeds should a second bounded experiment include an actual revisit to test loop closure.
Choose the shortest useful interval from the existing video, or a small set of keyframes spanning the revisit, and again produce the complete deliverable.
Do not label the final map loop-closed until the revisit constraint and resulting geometry have been inspected.
Before any generated multi-station run, separately complete and inspect one generated, stitched 360 panorama against source photos.

## Limits and capture implications

Omnidirectional video improves directional coverage and can help tracking retain features during turns.
It cannot reveal occluded surfaces, provide depth from pure rotation, or recover detail absent from the source.
Our AIST video is only 1920 by 960 pixels, with known missing source regions.
Generating more crops or upscaling those frames adds no new observations.

Stitched dual-lens panoramas are only approximately central, particularly near close objects and stitching seams.
COLMAP's own panorama implementation calls out potential stitching artifacts when applying exact rig constraints.
If future captures retain calibrated original fisheye images, modeling the physical cameras is an alternative to test.
It is not available from the current stitched AIST MP4 alone.

For new property capture, prioritize slow translation, sharp exposure, coverage around furniture, doorway transitions, and a return through previously seen areas.
Record at least one known physical length if the app needs metric dimensions; monocular reconstruction alone leaves global scale ambiguous.
Several independent measurements would be needed to assess dimensional accuracy rather than merely set scale.

## Verified local readiness

Read-only checks during this research found COLMAP 4.2.0 installed without GPU support.
The upstream matching-version panorama example includes `--use_cpu`.
`pycolmap` is not installed in `.venv-reconstruction`, so the example is not currently runnable there without setup.
Existing local tools include the native OpenMVS build, the completed LingBot baseline, and the captured-station viewer.
Neither stella_vslam nor the newly reviewed learned panorama methods was installed or run during this research.

Metashape's [system requirements](https://www.agisoft.com/downloads/system-requirements/) list Apple processors and GPU acceleration.
Its performance and spherical reconstruction quality on our AIST clip remain untested.
The commercial application is a useful external comparator, not evidence that the open-source recommendation already meets the fidelity requirement.

This research continues [issue #8](https://github.com/zeeshanhaque21/lingbot-map/issues/8).
The next implementation decision is the bounded calibrated-camera and OpenMVS experiment, not a full walkthrough or a new generative batch.
