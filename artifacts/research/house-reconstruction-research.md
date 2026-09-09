# House reconstruction from walkthrough video

Implementation continuation, 2026-09-09: the [native Mac pipeline and measured failures](mac-reconstruction-results.md) now include complete walkthrough reconstruction and exported-asset checks.
The latest [image-constrained depth and Brush experiments](photometric-and-brush-results.md) document failed depth corrections and successful native Metal training of a local Gaussian model.
The [Gaussian surface experiments](gaussian-surface-results.md) and [native OpenMVS pipeline](openmvs-mac-results.md) continue this work through actual triangle meshes, complete textured GLB exports and reserved-view checks.
Neither establishes a verified complete property model.
The local appearance improvement is not a verified full-building reconstruction.

## Recommendation

Start with **LingBot-Map on the existing video**, evaluate Depth Anything 3 as a complementary reconstruction/depth candidate, and retain a conventional photogrammetry baseline.
Do not select a depth network as the entire house-scanning architecture.
For reliable dimensions in a future product, prefer a capture mode that records calibrated camera poses and LiDAR depth, with independent measured checks.
Produce a measurement-oriented mesh and a photorealistic viewing layer separately.

The strongest discovery is unusually specific: the supplied `indoor_travel.MP4` is byte-for-byte identical to the indoor walkthrough distributed in the official LingBot-Map demonstration dataset.
The published file hash matches the local SHA-256, and the upstream repository provides a worked reconstruction example for that filename.[^1][^2]
This makes LingBot-Map the most relevant first experiment, but not a proven winner on dimensional accuracy.
A demonstration selected by its authors is also not an independent generalization test.

A community Apple Silicon/MPS port exists.
Its author reports successful, slow reconstruction, but its interface displays the interactive scene after processing rather than proving live cloud generation during inference.[^3]
Its published measurements are not measurements on this Mac.
The appropriate next step is a small, controlled compatibility and geometry test, not a full application build.

The working assumptions are that both a realistic walkthrough and useful dimensions matter, the existing MP4 is the first input, Apple Silicon is preferred, and improved capture can be considered later.
An Apple-only production requirement, survey-grade tolerance, or prohibition on new capture would materially change the recommendation.

## 1. Input assessment

| Property | Local observation |
|---|---|
| Input | `/Users/zeeshanhaque/Downloads/indoor_travel.MP4` |
| Video duration | 500 seconds, or 8 minutes 20 seconds |
| Encoded image size | 3840 × 2160 |
| Codec and nominal frame rate | HEVC, 50 frames/second |
| Frame count | 25,000 |
| File size | 5,619,812,145 bytes |
| SHA-256 | `ca590d8af33c465953d6ed6c305557e4b4d7cc6ba3710b4c65872bbf06b684e0` |
| Visual inspection | 13 stills, at 40-second intervals from 0 to 480 seconds |

The sampled scenes appear to be a furniture-store walkthrough, rather than a house.
They contain glossy floors, glass cabinets, repeated furniture and shelving, people, and some blurred frames.
These observations make it a useful difficult indoor example, but do not establish complete surface coverage, successful revisits, or recoverable camera calibration.
Only 13 of 25,000 frames were inspected visually.
No reconstruction, continuous tracking assessment, or dimensional measurement has been performed.

The file metadata inspected does not establish usable calibrated camera poses or depth measurements.
Such data must not be assumed simply because the file originated from a mobile camera.
The official LingBot-Map README describes its approximately 25,000-frame example as a 13-minute walkthrough, which differs from the matching local file's measured 500-second duration.[^2]
Processing and sampling should use the file timestamps, not the prose description.

The hash match is stronger evidence than a similar filename or screenshot.
It establishes that an existing published workflow targets this exact input.
It does not establish that the published model has correct dimensions, that every surface is recovered, or that the Apple port reproduces the original output.

## 2. What “similar to Matterport” requires

A house model has at least three independently testable outcomes.
A visually convincing walkthrough lets someone look around from new viewpoints.
A coherent reconstruction places rooms, doorways, walls, and repeated observations in one consistent coordinate system.
A measurement-oriented model additionally has validated scale and bounded geometric error.

A system can succeed at the first outcome and fail at the other two.
Soft, overlapping points or Gaussian splats can hide small pose errors from selected viewpoints.
Conversely, a geometrically sound wall mesh can look unattractive without textures and view-dependent appearance.
These should be separate acceptance categories and, where useful, separate deliverables.

Matterport's Pro3 specification gives an approximately 20 mm raw depth accuracy at 10 m.[^4]
That is a sensor specification, not a promise that every dimension in an entire reconstructed building has that error.
It also comes from a substantially different capture system than an uncalibrated RGB walkthrough.
It would be misleading to promise comparable accuracy from an ordinary MP4 on the strength of a depth-model benchmark.

With monocular imagery alone, geometric reconstruction does not inherently establish absolute scale.
A learned model can supply a metric prior, but a plausible learned scale is not a measurement certificate.
At least one trustworthy scale reference is needed for calibrated dimensions, and additional independent references are needed to test whether that scale remains consistent across the building.
No algorithm can recover the true geometry of rooms or wall surfaces that were never observed.

The principal risks are therefore camera-path drift, incorrect matching of similar rooms, scale inconsistency, reflective or transparent surfaces, moving objects, insufficient parallax, and incomplete coverage.
Improving per-frame depth can help with some of these risks, but does not eliminate them.

## 3. Candidate comparison

| Candidate | Appropriate role | Apple status supported by the reviewed evidence | Main qualification |
|---|---|---|---|
| LingBot-Map | First video reconstruction experiment | Community MPS port with author-reported runs | Exact supplied video has an official worked example; dimensional accuracy remains unverified |
| Depth Anything 3 | Multi-view geometry and dense-depth candidate | CoreML variants exist, but full long-video Mac pipeline is not established here | Use the appropriate multi-view model and alignment pipeline, not independent monocular depth |
| MASt3R-SLAM | Tracking and global-consistency comparator | Documented implementation targets CUDA | Explicit tracking, loop closure, and optimization are relevant to whole-building consistency |
| COLMAP | Conventional camera-pose and sparse-geometry baseline | macOS installation documented; dense GPU pipeline has separate backend requirements | Useful independent baseline; indoor textureless/repeated surfaces remain difficult |
| MapAnything | Secondary multi-view/metric reconstruction comparator | No complete MPS workflow validated here | Broader geometric inputs, but not a drop-in live house scanner |
| Metashape | Commercial photogrammetry baseline | Vendor documents native Apple Silicon GPU acceleration | Mature reconstruction alternative; still needs overlap and scale control |
| ARKit + RoomPlan | Preferred new-capture foundation | Native supported Apple capture APIs on appropriate devices | Cannot retrofit synchronized LiDAR and AR tracking into the existing MP4 |
| Brush | Optional photorealistic splat training/viewing | Native Mac support documented | Requires reconstructed camera data; not a substitute for accurate geometry |

### LingBot-Map

LingBot-Map is designed for streaming reconstruction, using geometric context and camera-history mechanisms rather than treating frames independently.
Its official repository includes a windowed long-video workflow and a worked `indoor_travel.MP4` example using `demo_render/batch_demo.py` and `demo_render/config/indoor.yaml`.[^2]
The model card distinguishes checkpoint variants, including a long-sequence checkpoint.[^5]

The original accelerated stack uses CUDA-oriented components, including FlashInfer, with a PyTorch SDPA alternative for attention.
The batch rendering stack also has its own dependencies.
Replacing attention alone does not establish that the renderer and full application work on Apple hardware.[^5]

The community `lingbot-desktop-mac` repository supplies an actual patch set and pins its upstream revision.
Its README reports approximately 0.3–1.0 processed frames/second generally, with three example runs of 16, 237, and 100 frames on the author's stated M5 Max machine.
It switches longer inputs to small-window processing and describes the output explicitly as a point cloud, not trained Gaussian splatting.[^3]
These are promising compatibility reports, not independently reproduced performance or accuracy results.

Long-term consistency is still a test requirement.
An open upstream issue reports drift on long indoor revisits and asks about loop closure or global optimization.[^6]
That single report does not prove universal failure, but prevents treating an impressive demonstration as sufficient evidence for reliable house measurements.
The model's learned drift-correction mechanisms should not be casually equated with a verified global optimization workflow.

### Depth Anything 3

DA3 remains a serious candidate.
Its multi-view models estimate depth and camera-related geometry, and its streaming extension processes overlapping chunks with alignment and loop-closure support.[^7][^8]
This is substantially closer to reconstruction than producing an independent depth map for each video frame.

The documented streaming memory figures depend strongly on resolution and chunk size.
For example, its reported TUM configuration at 504 × 378 uses 18.7 GB at a 30-frame chunk and 28.3 GB at a 120-frame chunk.
The smaller KITTI image shape yields different numbers.[^8]
These are upstream configurations, not measured MPS requirements, and should not be summarized as a universal “runs under 12 GB” claim.

The streaming source's long-sequence workflow performs chunk inference and alignment work before its final aligned point-cloud export stage.[^9]
Existing per-chunk output files therefore do not by themselves provide live cloud generation during inference.
A live viewer needs an earlier publication point and a way to revise transforms after alignment changes.

DA3 should be tested both as an end-to-end multi-view candidate and, if another method produces stronger camera poses, as a depth contributor conditioned on those poses where the selected API supports it.
The second approach avoids requiring one model to be best at tracking, scale, dense depth, and final surface reconstruction simultaneously.
License selection is checkpoint-specific: the official model table distinguishes permissive small/base variants from noncommercial larger variants.[^7]

### MASt3R-SLAM and COLMAP

MASt3R-SLAM combines learned matching with camera tracking, map construction, loop closure, and optimization.[^10]
That makes it a useful comparator when local depth looks convincing but rooms bend, duplicate, or fail to align on revisits.
Its documented setup and performance experiments are CUDA-oriented, so Mac compatibility must not be assumed from the underlying use of PyTorch.
It is a conditional second-line evaluation, not a reason to begin a substantial MPS port before checking geometry quality.

COLMAP supplies a conventional structure-from-motion baseline with camera estimation, bundle adjustment, and sparse reconstruction.
Its documentation discusses sequential matching and loop-detection strategies relevant to video.[^11]
This baseline tests whether learned reconstruction is adding value beyond established image geometry, and can provide poses for subsequent depth fusion.

macOS installation is documented, but sparse reconstruction support and dense GPU reconstruction support are separate matters.[^12]
The plan should not promise an all-Metal dense COLMAP pipeline.
Uniformly extracting all 25,000 frames is also not a sensible default: many adjacent views are redundant, while difficult turns may need higher temporal density.

### MapAnything and Metashape

MapAnything accepts different combinations of images and geometric information and provides multi-view reconstruction and downstream export workflows.[^13]
It is worth retaining if the first comparisons expose a specific failure involving camera assumptions or metric reconstruction.
Its broad capabilities do not establish low-memory streaming or an Apple-compatible end-to-end pipeline.
Avoid adding it to the first full-video comparison merely to increase the number of models tested.

Metashape is the clearest commercial Mac alternative in the shortlist.
Agisoft's current hardware guidance explicitly documents native Apple Silicon support, including GPU acceleration.[^14]
Its API documentation includes a video-import task, and its scale-bar guidance recommends multiple control distances rather than relying only on a single reference.[^15][^16]
It is useful as an off-the-shelf reconstruction baseline, not evidence that arbitrary indoor video will produce survey-quality geometry.

## 4. Apple execution and earlier repository questions

Three different meanings of “Metal support” need to remain separate.
PyTorch MPS is a tensor execution backend.
CoreML is a model deployment runtime that can choose among allowed compute resources.
Metal rendering displays points or surfaces and can work even when reconstruction runs somewhere else.
Evidence for one is not evidence for all three.

An explicit `mps` device option is evidence of intended support, not a completed compatibility test.
A useful verification must establish model execution, supported operators, synchronization and readback correctness, finite outputs, memory behavior, and reconstruction quality on fixed inputs.
Likewise, permitting CPU and GPU execution in CoreML is not proof that every operation executes on the GPU.[^17]

The `mlboydaisuke/Depth-Anything-3-Base-CoreML` package is relevant to Apple depth inference, but is not evidence of a complete long-video mapping system.[^18]
The earlier `VideoDepthViewer3D` candidate is a depth-video visualization starting point, not demonstrated globally registered house reconstruction.[^19]
Animating each frame's depth surface is different from placing all frames in a common world coordinate system.

The existing small-view iPhone DA3 approach also should not be extrapolated directly to hundreds or thousands of observations.
Long-sequence state management, tracking recovery, revisits, scale consistency, and surface reintegration are separate engineering and validation problems.
For the current task, the published LingBot MPS patch is a more concrete starting point than assuming an arbitrary CUDA pipeline will run after changing a device string.

For licensing, distinguish project code, model weights, and inherited components.
LingBot's upstream materials identify Apache 2.0, while the Mac wrapper separately cautions about checkpoint terms and model lineage.[^3][^5]
Lineage alone does not establish the checkpoint's legal terms.
Before commercial distribution, archive and review the actual selected checkpoint's license and notices, and resolve any ambiguity with its publisher.

## 5. Accurate geometry and a live preview

The proposed system should retain camera observations and a revisable map, rather than permanently appending every predicted point into a single unstructured cloud.
Each accepted frame or chunk needs its timestamp, calibrated or estimated intrinsics, pose, depth, color association, quality information, and a stable identifier.
Global alignment can then revise the placement of earlier observations without losing their provenance.

The viewer should show new provisional geometry as reconstruction runs.
It should also show the camera path, the currently processed frame, accepted/rejected observations, and which geometry has been globally corrected.
When a loop correction changes old poses, the affected geometry must move or be rebuilt.
A progress bar followed by a finished viewer, or a playback animation of a completed cloud, does not satisfy the incremental-generation requirement.

For a surface model, TSDF integration is a reasonable first candidate once poses and depth are consistent.
Open3D documents integrating RGB-D observations using camera parameters and extracting a point cloud or triangle mesh.[^20]
Its reconstruction workflow also describes integrating observations using optimized poses.[^21]
The relevant design implication is to preserve inputs so that corrected camera trajectories can drive reintegration.

Fusion is not a cure for wrong geometry.
Averaging contradictory depths from wrong poses can thicken walls and conceal the source of the error.
Reflective floors, glass, moving people, uncertain depth boundaries, and inconsistent observations should be rejected or downweighted.
Model confidence scores should not be presented as calibrated accuracy probabilities without validation.
Choosing a 2 cm voxel does not establish 2 cm reconstruction accuracy.

The optional appearance layer should follow geometry validation.
Brush supports Mac execution and uses COLMAP or Nerfstudio-style datasets for Gaussian-splat training and interactive viewing.[^22]
It can improve the walkthrough experience, but the measurement interface should remain attached to validated surfaces.
Converting points into a splat file format is not the same operation as optimizing Gaussian splats against source photographs.

## 6. New capture for dependable house dimensions

For future recordings, the strongest Apple-native path is to capture RGB, camera intrinsics, ARKit poses, timestamps, and available scene depth/confidence together.
Supported LiDAR devices expose scene depth through ARKit, and Apple provides an example that unprojects depth into a world-space point cloud.[^23][^24]
These observations provide measured depth and tracking information that an ordinary exported MP4 does not supply.

RoomPlan provides a complementary structured representation of room elements and dimensions.
Its multiroom APIs support combining room captures into a structure.[^25][^26]
This is valuable for walls, openings, and an interpretable floor-plan layer, but should not be confused with a fully textured, photorealistic surface model.

Capture quality remains important even with LiDAR.
Walk with translation and overlap, observe doorways from both sides, revisit connecting areas, avoid rapid turns, and inspect missing coverage before leaving.
Record tracking interruptions and preserve floor or room transitions explicitly.
For multiple floors, do not silently treat every room as lying on one plane.

Use measured reference distances distributed across the property, with some reserved for independent checking.
RoomPlan dimensions and ARKit poses are useful inputs, not substitutes for validation.
If the application needs contractual or survey-grade accuracy, its acceptance test must be tied to an appropriate independent measurement method and capture protocol.
The existing demo clip cannot establish that level of accuracy by itself.

## 7. Phased implementation proposal

These phases are a proposal, not completed implementation.
Only input inspection, source research, and this report are complete.
Each implementation phase should begin by reading the cited documentation at a pinned revision and recording the exact supported arguments and data conventions.

### Phase 0: Freeze documentation and evaluation scope

**Work:** Record the supplied video hash, chosen repository revisions, checkpoint identifiers and hashes, licenses, preprocessing rules, and evaluation clips.
Resolve differences between the LingBot GitHub and model-card examples before choosing window and keyframe settings.
The published demo is an implementation reference, not ground truth.

**Documentation and allowed entry points:** Use the LingBot README's `demo.py`, `demo_render/batch_demo.py`, and `demo_render/config/indoor.yaml` patterns; the Mac wrapper's pinned upstream revision and patch; DA3's streaming README; and COLMAP's tutorial.[^2][^3][^8][^11]
For later phases, use documented Open3D integration examples, Brush's supported dataset layout, and Apple's scene-depth and multiroom examples.[^20][^22][^24][^26]
These are verified documentation entry points, not permission to invent a common `stream()` API across repositories.

**Verification:** Produce a manifest and a fixed set of representative segments, including clear translated motion, a turn, reflective surfaces, and a moving-person interval.
Confirm timestamp-to-frame mappings and identify actual revisits by reviewing the continuous footage.

**Guards:** Do not infer missing camera calibration, scale references, or loops from filenames and isolated stills.
Do not silently overwrite existing project files or install through an unreviewed shell bootstrap.

### Phase 1: Small Apple compatibility and reconstruction test

**Work:** Reproduce the Mac port's documented short-input workflow in an isolated environment, beginning with tens of frames before a longer representative segment.
Preserve the patch provenance and inspect differences from current upstream, especially attention-cache fixes.
Measure actual model device, output validity, memory, wall time, and camera/point-cloud consistency.

**Documentation:** Follow the port's manual installation and patch structure rather than merely changing `cuda` to `mps`.[^3]
If a CPU or CUDA reference is available within the approved resource scope, compare the same checkpoint and preprocessing on identical small inputs.
Such a comparison is useful for finding backend differences, not for establishing real-world dimensional accuracy.

**Verification:** Show finite, correctly oriented geometry with a plausible trajectory, reproducible outputs within expected numeric variation, and bounded memory on the selected window configuration.
Repeat the timed segment at least three times, separating initialization from steady-state inference and export.
Record failures as failures rather than enabling an undocumented CPU fallback and retaining an MPS label.

**Guard and decision gate:** Do not estimate whole-video runtime from an unrelated author's machine.
If the port fails correctness checks, decide whether to use a documented CUDA reference or a commercial Mac baseline before undertaking a substantial backend rewrite.

### Phase 2: Compare geometry before building the product

**Work:** Run LingBot and a conventional COLMAP sparse baseline on the same selected observations.
Add DA3 on representative overlapping windows, then test whether its complete reconstruction or its depth with stronger external poses offers a useful improvement.
Use MASt3R-SLAM if tracking and revisit consistency remain the dominant problem and an appropriate CUDA environment is authorized.
Retain MapAnything as a targeted follow-up, not a mandatory fifth full-video run.

**Documentation:** Copy the documented matching, reconstruction, and export workflows for each candidate.[^8][^10][^11][^13]
Keep intrinsics, image resizing, masks, and coordinate conventions explicit.

**Verification:** Compare registration coverage, duplicated walls, room connectivity, cross-view depth disagreement, visible revisit alignment, and failures on moving or reflective surfaces.
Use the untouched source video as a do-nothing presentation baseline: a 3D result should add useful navigation without implying measured geometry that has not been validated.
Use unmodified COLMAP as the algorithmic baseline so gains are not attributed solely to a new viewer or different frame selection.

**Guard and decision gate:** No absolute dimensional accuracy claim is permitted for this clip without independent references.
Do not rank candidates solely by attractive screenshots or one camera-trajectory score.
Select a reconstruction backbone only after reviewing matched regions and failure cases.

### Phase 3: Full-sequence processing and actual incremental viewing

**Work:** Introduce adaptive keyframe selection and bounded-window processing for the full video.
An initial 2–5 source frames/second would yield 1,000–2,500 candidate observations over this file, but this is an experiment setting, not a processing-speed forecast.
Increase temporal sampling around turns and difficult tracking intervals if necessary.
Keep original-resolution images for texture work rather than requiring the geometry model to ingest 4K frames.

Publish provisional chunk geometry while inference is running, with stable IDs and revisable poses.
Persist accepted observations and resumable state as processing advances.
Use level of detail and bounded display density so the viewer does not become the memory bottleneck.

**Documentation:** Start from LingBot's prediction-saving and viewer examples, or DA3's streaming processing/export boundaries, depending on the Phase 2 winner.[^2][^9]
The publication hook is new integration work and must be located in the actual selected code version before implementation.

**Verification:** Demonstrate new world-space geometry appearing before the final frame is processed.
Show an earlier chunk moving correctly after an alignment update, with no duplicate stale geometry.
Interrupt and resume a test run without losing completed observations.
Report preview latency separately from total reconstruction time.

**Guards:** Do not label prerecorded playback “live reconstruction.”
Do not retain every dense frame cloud indefinitely or permanently fuse provisional geometry without a correction strategy.

### Phase 4: Surface, appearance, and house deliverables

**Work:** Integrate accepted depths with optimized poses, then extract a mesh.
Keep observed geometry distinct from any hole filling or fitted structural approximation.
Add textures or optional Brush-based splats only after geometric checks pass.
Generate structural walls or a floor plan only where observations support them.

**Documentation:** Follow Open3D's integration and optimized-pose reintegration examples and Brush's supported dataset import workflow.[^20][^21][^22]
For a separate new-capture branch, follow Apple's ARKit point-cloud example and RoomPlan structure-building documentation.[^24][^26]

**Verification:** Export and reopen a colored point cloud and mesh in an independent viewer.
Check units, axes, handedness, timestamps, camera-to-world conventions, and texture registration.
Keep a processing manifest alongside the outputs.
Verify that measurements refer to the intended surface model, not an appearance-only splat representation.

**Guard:** Do not fabricate hidden rooms, ceiling geometry, or wall thickness and present them as measured.
Do not prioritize cosmetic smoothing over preserving known dimensions.

### Phase 5: Independent house validation and final selection

**Work:** Capture a separate house or apartment with reference measurements, preferably including LiDAR/ARKit data.
The official demo video remains a reproducibility fixture, while this independent site tests the actual product objective.
Use one subset of references for scale or alignment and a disjoint subset for evaluation.

**Verification:** A proposed initial acceptance target is error no greater than the larger of 5 cm or 2% on at least 90% of 10–20 held-out structural distances distributed across rooms and directions.
This is a provisional product target, not an achieved result, a universal safety threshold, or a Matterport specification.
Report median, worst case, sample count, room coverage, and every failure rather than only the pass percentage.
Add checks for false room merges, missing connections, duplicate walls, and explicitly identified unobserved areas.

Where reference poses or surfaces exist, report appropriate trajectory and surface comparisons as additional metrics.
If scale is fitted during alignment, disclose that operation and separately report scale error before fitting.
A similarity-aligned trajectory score can hide the very scale problem that matters for house measurements.

**Final audit:** Recheck implementation against pinned documentation, output conventions, selected checkpoint terms, and the proposed no-fabrication and no-hidden-fallback guards.
Only after this gate should the result be described as dimensionally validated.
If video-only reconstruction misses the target, prioritize richer capture rather than repeatedly swapping depth models without changing the missing information.

## 8. Decision and remaining uncertainty

The recommended immediate experiment is LingBot-Map on a short segment of the exact supplied video, using the community MPS patch as a candidate implementation and the official example as a reference.
The next comparison is DA3 and conventional image geometry on matched observations.
The recommended long-term capture architecture is ARKit/LiDAR plus independently checked geometry, with RoomPlan as a structural layer and optional splats or textures for presentation.

The remaining uncertainties are material: no model has been run locally for this report; the Mac port's performance is author-reported; global geometry has not been checked; and no independent dimensions are available for the existing clip.
Published demo success establishes feasibility of producing an output, not the accuracy of that output.
No installation, model download, reconstruction run, or application implementation is part of this report.

## Sources

The following primary documentation and project-maintainer sources were reviewed for this report.
Mutable repository pages describe the versions available during the September 2026 review; implementation should pin exact commits and checkpoints.
Community reports are identified as such and are not treated as independently reproduced tests.

[^1]: Robbyant, [LingBot-Map demo dataset: indoor_travel.MP4 file metadata and SHA-256](https://huggingface.co/datasets/robbyant/lingbot-map-demo/blob/main/indoor_travel.MP4), undated dataset file page. Compared with the local file hash and metadata.
[^2]: Robbyant Team, [LingBot-Map README, worked indoor example and streaming modes](https://github.com/Robbyant/lingbot-map), 2026, mutable repository documentation.
[^3]: donalleniii / Asteria Research and Technology, [Lingbot-Desktop Mac README](https://github.com/donalleniii/lingbot-desktop-mac), 2026 community port, including manual installation, patch provenance, output limitations, and author-reported tests.
[^4]: Matterport, [Pro3 camera specifications](https://matterport.com/store/product/cam-mc300-x), current product page, raw depth accuracy specification.
[^5]: Robbyant Team, [LingBot-Map model card](https://huggingface.co/robbyant/lingbot-map), 2026, checkpoint variants, execution dependencies, and license statement.
[^6]: LingBot-Map community, [Issue 78: long-indoor drift discussion](https://github.com/Robbyant/lingbot-map/issues/78), June 2026, individual user report rather than a controlled benchmark.
[^7]: ByteDance Seed, [Depth Anything 3 README and model table](https://github.com/ByteDance-Seed/Depth-Anything-3), current repository documentation, multi-view capabilities and checkpoint licenses.
[^8]: ByteDance Seed, [DA3 Streaming README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md), current documentation, overlapping chunks, alignment, memory measurements, and outputs.
[^9]: ByteDance Seed, [DA3 long-sequence streaming implementation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/da3_streaming.py), current source, order of inference, alignment, and point-cloud export.
[^10]: MASt3R-SLAM authors, [MASt3R-SLAM repository](https://github.com/rmurai0610/MASt3R-SLAM) and [project page](https://edexheim.github.io/mast3r-slam/), original implementation and research description, 2024–2025.
[^11]: COLMAP authors, [COLMAP tutorial](https://colmap.github.io/tutorial.html), current documentation, matching, sparse reconstruction, and dense reconstruction stages.
[^12]: COLMAP authors, [Installation documentation](https://colmap.github.io/install.html), current documentation, macOS installation and GPU backend requirements.
[^13]: Meta and MapAnything authors, [MapAnything repository](https://github.com/facebookresearch/map-anything), current model, supported geometric inputs, export workflows, and checkpoint variants.
[^14]: Agisoft, [Hardware recommendations](https://agisoft.freshdesk.com/support/solutions/articles/31000172512-hardware-recommendations), updated July 16, 2026, native Apple Silicon and GPU acceleration support.
[^15]: Agisoft, [ImportVideo API reference](https://download.agisoft.com/metashape-java-api/latest/com/agisoft/metashape/tasks/ImportVideo.html), current task documentation.
[^16]: Agisoft, [Creating scale bars without coded targets](https://agisoft.freshdesk.com/support/solutions/articles/31000162602-creating-scale-bars-in-the-project-without-coded-targets), 2026 helpdesk guidance, scaling and multiple reference distances.
[^17]: Apple, [MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits), current CoreML documentation, compute-unit choices.
[^18]: mlboydaisuke, [Depth-Anything-3-Base-CoreML](https://huggingface.co/mlboydaisuke/Depth-Anything-3-Base-CoreML), community model package and model card.
[^19]: amariichi, [VideoDepthViewer3D](https://github.com/amariichi/VideoDepthViewer3D), repository implementation and documentation, depth-video visualization candidate.
[^20]: Open3D contributors, [TSDF integration](https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html), current documentation, RGB-D integration and surface extraction.
[^21]: Open3D contributors, [Integrate scene](https://www.open3d.org/docs/latest/tutorial/reconstruction_system/integrate_scene.html), current reconstruction-system documentation, fusion with optimized poses.
[^22]: Arthur Brussee and contributors, [Brush](https://github.com/ArthurBrussee/brush), current repository documentation, supported platforms, datasets, and Gaussian-splat training/viewing.
[^23]: Apple, [ARFrame.sceneDepth](https://developer.apple.com/documentation/arkit/arframe/scenedepth), current ARKit documentation.
[^24]: Apple, [Displaying a point cloud using scene depth](https://developer.apple.com/documentation/arkit/displaying-a-point-cloud-using-scene-depth), official capture and rendering example.
[^25]: Apple, [RoomPlan](https://developer.apple.com/augmented-reality/roomplan/), official framework overview.
[^26]: Apple, [Scanning the rooms of a single structure](https://developer.apple.com/documentation/roomplan/scanning-the-rooms-of-a-single-structure) and [StructureBuilder](https://developer.apple.com/documentation/roomplan/structurebuilder), official multiroom documentation.

## Implementation continuation - 2026-09-09

A native Mac pipeline is now implemented in `lingbot_map/reconstruction/`.
See [the usage guide](../../RECONSTRUCTION.md) and [the Lucida/object reconstruction follow-up](lucida-and-object-reconstruction.md).
The implementation combines Metal inference, aligned photogrammetry rotation groups, a depth-constrained translation graph, adaptive dense motion bridges, observed-surface fusion and source-view validation.
A checkpoint-specific camera convention mismatch was identified experimentally and corrected at the adapter boundary.
The [implementation report](mac-reconstruction-results.md) records completed runs, failed alternatives, numerical checks and unresolved acceptance requirements.
Measured building dimensions remain unverified without external references.
The later [FixAnything assessment](fixanything-assessment.md) distinguishes rendered-video refinement from verified mesh reconstruction.
