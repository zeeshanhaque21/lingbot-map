# Backlog

## In flight
- [ ] mac-property-reconstruction - Implement and validate faithful Mac video-to-mesh reconstruction (since 2026-09-09)
  The native Mac reconstruction pipeline is implemented, with experimental Brush and OpenMVS paths, but full-building fidelity remains unresolved.

  Current evidence: 48 tests and Ruff pass.
  The Cauchy full-walkthrough candidate still fails five app views and six full-mesh views.
  Higher-resolution Brush improves common-resolution RGB PSNR from 21.69 to 23.12 dB across five reserved images, but its extracted app mesh has only 61.92% median visible coverage.
  Native CPU OpenMVS reconstructs 377,940 dense points and exports a complete 472,011-triangle textured GLB.
  With seam adjustments disabled, that local mesh has 88.06% median coverage and 0.07959 visible RGB error, but fails the Gaussian-depth agreement screen and retains visible surface defects.
  Source hashes, failed attempts, texture packaging checks and a dense/sparse handoff spike are preserved.

  Native photometric refinement completes but lowers local coverage and whole-image PSNR.
  The complete 900-training-image native reconstruction is running, with 100 captured views reserved for evaluation.
  Multi-material packaging and complete-camera evaluation are implemented; 1,000 viewer poses and timestamps match the calibrated capture.
  Global camera paths disagree beyond one similarity, so building layout remains unverified.
  Next: finish and inspect the full native export, evaluate every reserved view, and resolve the global camera-path and connectivity inconsistencies.
  Actual property capture, measured scale and independent dimensions remain pending.
  Do not mark listing readiness true from these internal checks.

  Draft PR: https://github.com/zeeshanhaque21/lingbot-map/pull/2
  Issues: #1 and #3.

  User supplied example/ sequences: loop 237, courthouse 286, university 324 RGB images, all 518 by 294.
  Direct lossless image import and evaluation are tracked in GitHub issue #4.
  Full native geometry completed with 9,107,943 faces.
  Original texture packing remains CPU-active; local virtual-face grouping increases patches and visible RGB error.
  A separate unchanged-geometry dense-point-color export completes in 6.16 seconds, with all 100 reserved views being evaluated.
  All 49 tests pass after the image-input change.

## Queued
## Done
