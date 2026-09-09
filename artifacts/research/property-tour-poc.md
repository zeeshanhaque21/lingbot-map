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
The full 61-frame, 10-step run at 832 by 480 is in progress.
Next acceptance work is spherical stitching, overlap and seam comparison against the raw baseline, and visual inspection of preserved structural details.

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
