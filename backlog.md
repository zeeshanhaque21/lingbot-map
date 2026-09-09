# Backlog

## In flight
- [ ] mac-property-reconstruction - Implement and validate faithful Mac video-to-mesh reconstruction (since 2026-09-09)
  Native Mac pipeline implemented; faithful geometry remains unresolved. Full SE3 and direct bundle experiments completed for 1000 cameras. Cauchy refinement lowers reserved feature error from 1.614 to 0.576 pixels across 122264 correspondences; its app mesh fails five of 100 views and full mesh fails six. MapAnything Apache runs locally on MPS with strict local checkpoint loading. Four 48-frame variants, including calibration, pose conditioning and fixed registered cameras, complete export but fail geometry screening. 27 tests and Ruff pass. Next: direct multiview depth correction with fixed cameras and explicit occlusion handling. Actual property capture and measured scale requested for final verification. Draft PR https://github.com/zeeshanhaque21/lingbot-map/pull/2; fidelity issue https://github.com/zeeshanhaque21/lingbot-map/issues/3.
## Queued
## Done
