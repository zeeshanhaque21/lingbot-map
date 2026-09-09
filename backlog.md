# Backlog

## In flight
- [ ] mac-property-reconstruction - Implement and validate faithful Mac video-to-mesh reconstruction (since 2026-09-09)
  The native Mac reconstruction pipeline is implemented, with experimental Brush and OpenMVS paths, but full-building fidelity remains unresolved.

  Current evidence: 47 tests and Ruff pass.
  The Cauchy full-walkthrough candidate still fails five app views and six full-mesh views.
  Higher-resolution Brush improves common-resolution RGB PSNR from 21.69 to 23.12 dB across five reserved images, but its extracted app mesh has only 61.92% median visible coverage.
  Native CPU OpenMVS reconstructs 377,940 dense points and exports a complete 472,011-triangle textured GLB.
  With seam adjustments disabled, that local mesh has 88.06% median coverage and 0.07959 visible RGB error, but fails the Gaussian-depth agreement screen and retains visible surface defects.
  Source hashes, failed attempts, texture packaging checks and a dense/sparse handoff spike are preserved.

  Next: test native photometric mesh refinement against the unchanged OpenMVS baseline, then establish full-walkthrough geometry and room connectivity.
  Actual property capture, measured scale and independent dimensions remain pending.
  Do not mark listing readiness true from these internal checks.

  Draft PR: https://github.com/zeeshanhaque21/lingbot-map/pull/2
  Issues: #1 and #3.

## Queued
## Done
