# Rough property tour PoC

The current product target is a cleaned rough mesh with loop constraints, clickable stations and refined 360 views.
Independent dimensions and photorealistic mesh textures are outside this PoC's acceptance criteria.
Doors, windows and room connectivity still need to agree with the captured property.

## Local artifact

The live viewer is available at `http://127.0.0.1:8083/` while the local process is running.
Its portable asset directory is `reconstructions/tours/loop-v1`.
The mesh has 150,000 triangles and occupies 3,121,596 bytes.
Six stations retain captured reference images, camera poses, raw panoramas, radial depth and observed-surface masks.
The viewer supports overview rotation, clickable mesh markers, panorama rotation, zoom, keyboard controls and station switching without external web dependencies.

The input is the supplied `example/loop` image sequence, with 237 frames preserved byte-for-byte.
All 847 supplied images across the three example directories were imported.
The courthouse hybrid and native pipelines both exported rough meshes; the university hybrid run stopped on 11.3% overlapping-depth disagreement.
The indoor loop is the current tour candidate.

## Loop and cleanup evidence

The loop experiment proposes 400 return-path pairs and retains 160 pairs with at least 15 geometrically verified feature matches.
The actual exported candidate cameras have median pair Sampson error 3.02 pixels, compared with 9.98 pixels for the unchanged baseline on the same matches.
The first-to-last pair has 32 matches and median error 2.25 pixels.
The audit supports a navigation connection across the return path without forcing different capture positions to coincide.
These are fitting residuals, not independent accuracy estimates.

Cleanup removes 8,765 tiny components and retains 3,279,602 triangles for panorama rendering.
Quadric simplification produces the navigation mesh.
The GLB roundtrip preserves the simplified vertex positions and triangle indices.
The full-resolution rendering mesh remains available separately.

## Spherical presentation

A 61-frame camera sweep samples seven latitude levels, including the ceiling and floor poles.
It starts and ends at the captured station view, with those two images supplied as clean FixAnything anchors.
The perspective renders use the same fixed center and calibrated intrinsics.
Spherical stitching uses the saved rotations and tapered overlap weights.

The raw sweep baseline covers 100% of spherical directions.
Mean overlap RGB standard deviation is 0.00075; wrap-edge RGB difference is 0.00195, both on a 0-to-1 scale.
Only 36% to 41% of raw panorama pixels intersect captured surfaces.
Coverage of rendering directions must not be confused with coverage of the actual building.

All model downloads completed through Motrix with checksum verification.
A real five-frame, two-step FixAnything smoke run completed on MPS in 104.17 seconds, including loading.
It produces plausible gap filling with visible artifacts and is not a final panorama.
The full 61-frame, 10-step run at 832 by 480 was stopped at the user's request after nine saved denoising checkpoints.
It produced no completed generated image set or stitched panorama.
Input inspection found 18 entirely empty renders; the downward smoke output also repeats the forward corridor instead of preserving its intended orientation.
Conditioning must be corrected before restarting generation.
One complete 360 station must pass generation, stitching, viewer inspection and source-photo comparison before additional stations or higher-cost runs.
The [conditioning investigation](fixanything-panorama-debugging.md) records the sampled evidence, author-method comparison and correction sequence.

The five actual smoke-test input/output pairs are visible at `http://127.0.0.1:8083/fixanything-smoke-v1/review.html`.
The gallery uses FixAnything's exact input crop and the unchanged generated PNGs.
The live 3D tour still displays raw panoramas until a full candidate has been reviewed.

A separate source-photo comparison is ready for all 24 images excluded from mesh fusion.
It projects shared surfaces through the fixed station using the saved mesh and cameras, rejects occluded surfaces, and compares both candidate and unchanged panorama against the same photo pixels.
The unchanged baseline has 1,197,452 shared pixels across those 24 views.
These photos still contributed to camera estimation and learned reconstruction context; color error is a diagnostic and does not certify building structure.
Synthetic translated-camera, occlusion, wraparound, pole and inference-input tests pass; the full suite has 57 passing tests.

## Reproduce

```bash
.venv-reconstruction/bin/python -m lingbot_map.reconstruction tour \
  --source reconstructions/examples/loop-closure-v1/final \
  --output reconstructions/tours/loop-v2
.venv-reconstruction/bin/python -m lingbot_map.reconstruction tour-view \
  --output reconstructions/tours/loop-v2
.venv-reconstruction/bin/python -m lingbot_map.reconstruction tour-sweep \
  --source reconstructions/tours/loop-v2 --station 000 \
  --output reconstructions/tours/loop-v2/sweep-000
.venv-reconstruction/bin/python -m lingbot_map.reconstruction tour-stitch \
  --source reconstructions/tours/loop-v2/sweep-000 \
  --images reconstructions/tours/loop-v2/sweep-000/raw-frames \
  --output reconstructions/tours/loop-v2/sweep-000/raw-stitch
```

Existing outputs are preserved; each export requires a fresh directory.
