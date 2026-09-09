# Backlog

## In flight
- [ ] mac-property-reconstruction - Implement and validate faithful Mac video-to-mesh reconstruction (since 2026-09-09)
  Native Mac pipeline implemented; fidelity issue #3 remains open. Full Cauchy GLB fails 5/100 reserved app views and 6/100 full-mesh views. Four MapAnything chair variants and three direct photometric depth corrections fail screening. Brush v0.3.0 runs on M3 Max Metal: COLMAP-seeded 48-frame trial trains 500000 Gaussians in 151.54 seconds, reaches 23.51 dB mean full-image PSNR on five reserved views versus 10.22 dB unchanged TSDF; weak chair view remains 18.33 dB. Local appearance evidence does not verify surface geometry or full-building scale. 39 tests pass. Next: geometric validation/surface recovery for shared Gaussians and complete-walkthrough consistency. Actual property capture and measured length requested, pending.
## Queued
## Done
