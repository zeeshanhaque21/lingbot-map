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
  Tests cover MPS precision, spherical projection, shared-surface reprojection and rejection of unsuitable spherical inputs.
  FixAnything and pinned DiffSynth source are installed in .venv-fixanything.
  The 82.86 GB model download completed with Motrix checksums.
  A five-frame, two-step smoke run completed in 104 seconds; its actual pairs are visible at http://127.0.0.1:8083/fixanything-smoke-v1/review.html.
  The full 61-frame, 10-step run was stopped at the user request after nine saved checkpoints; no full generated image set exists.
  Input inspection found 18 entirely empty renders and only one unique clean anchor image.
  The downward smoke output incorrectly repeats the forward corridor, so the smoke result cannot validate panorama stitching.
  The unchanged panorama was compared with all 24 source photos excluded from mesh fusion, covering 1,197,452 shared pixels.
  These photos still contributed to camera estimation and learned reconstruction context.
  Next: repair the input conditioning using the author-method research in fixanything-panorama-debugging.md.
  ALWAYS complete one representative 360 station end to end before any batch or higher-cost run.
  The pilot must generate every required view, stitch a complete panorama, work in the viewer and preserve observed layout in source-photo checks.
  A five-frame runtime smoke test does not count.
  The runner rejects the current unsupported sweep before model loading and respects all clean anchor indices in its manifest.
  Expose a refined candidate only with the raw panorama retained and review status explicit.
  Ego space 30 contains the raw tour, smoke gallery and sampled 61-view input inspection.
  The draft PR tracks the rough tour, failed conditioning evidence and the required complete single-station pilot.
  The native full indoor mesh and native TSDF comparisons remain research evidence rather than PoC acceptance requirements.

  Draft PR: https://github.com/zeeshanhaque21/lingbot-map/pull/2
  Product scope: https://github.com/zeeshanhaque21/lingbot-map/issues/5

  The LingBot quality diagnosis now includes a 72-frame stage ablation and an uninterrupted 237-frame context experiment.
  Reset-window raw closure error is 8.36 px; uninterrupted inference is 3.28 px; the current fitted cameras are 3.06 px on the same 153 pairs.
  Source-photo conditioning uses 24 photos in one complete 61-view input, but 18 directions remain empty and generation stays stopped.
  Projection worsens two of three reserved images slightly, so it remains experimental.
  Evidence: artifacts/research/lingbot-quality-diagnosis.md.

## Queued
## Done
