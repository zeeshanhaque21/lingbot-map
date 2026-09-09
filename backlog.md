# Backlog

## In flight
- [ ] mac-property-reconstruction - Build a rough 3D property tour with fixed 360 stations (since 2026-09-09)
  The user clarified the real estate product target: a rough, loop-closed and cleaned 3D model, with fixed locations linked to 360-degree views.
  Photorealistic mesh geometry and survey-grade dimensions are no longer the PoC acceptance target.
  The proposed appearance layer runs fixed-center rendered trajectories through FixAnything, then assembles spherical panoramas.
  The PoC must verify spatial navigation, loop closure, panorama seams and preservation of observed doors/windows/layout.
  Unknown surfaces remain explicit; generated appearance does not establish dimensions.
  Reuse the completed indoor-loop mesh and compare the additional verified loop-match candidate.
  Build station manifests, raw spherical renders and a navigable tour first, then test FixAnything locally.

  Implemented: direct image import, rough mesh export, six spherical stations and an interactive tour viewer at http://127.0.0.1:8083/.
  Indoor loop and courthouse meshes are exported; university remains rejected for an 11.3% window overlap disagreement.
  A loop candidate adds 400 end-to-start matching proposals with native geometric verification, retaining the baseline.
  The 3.1 MB navigation mesh is cleaned and simplified to 150,000 triangles.
  The return audit verifies 160 pairs; median pair error improves from 9.98 to 3.02 pixels.
  A 61-frame anchored spherical sweep and calibrated stitching baseline cover every direction.
  All 55 tests pass, including MPS precision, spherical projection and shared-surface reprojection checks.
  FixAnything and pinned DiffSynth source are installed in .venv-fixanything.
  The 82.86 GB model download completed with Motrix checksums.
  A five-frame, two-step smoke run completed in 104 seconds; its actual pairs are visible at http://127.0.0.1:8083/fixanything-smoke-v1/review.html.
  The full 61-frame, 10-step run is active with durable latent checkpoints.
  The unchanged panorama was compared with all 24 source photos excluded from mesh fusion, covering 1,197,452 shared pixels.
  These photos still contributed to camera estimation and learned reconstruction context.
  Next: finish the full Mac run, stitch it, inspect structural changes and seams, and run compare_panorama_sources.py against the generated panorama.
  Expose a refined candidate only with the raw panorama retained and review status explicit.
  Ego space 30 is handed to the user for the smoke gallery; fresh PR image rendering verification is pending browser availability.
  PR body is verified and image links remain unchanged and pinned to their previously verified commit.
  The native full indoor mesh and native TSDF comparisons remain research evidence rather than PoC acceptance requirements.

  Draft PR: https://github.com/zeeshanhaque21/lingbot-map/pull/2
  Product scope: https://github.com/zeeshanhaque21/lingbot-map/issues/5

## Queued
## Done
