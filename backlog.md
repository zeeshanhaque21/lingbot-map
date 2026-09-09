# Backlog

## In flight
- [ ] mac-property-reconstruction - Implement and validate faithful Mac video-to-mesh reconstruction (since 2026-09-09)
  Native Mac pipeline implemented; 17 tests and the 80-frame CLI integration pass. The hybrid 1,000-camera model resolves the kitchen occluder but seven app views fail the full 100-view audit. Uninterrupted native inference completed in 25.1 minutes; its app mesh fails 92 of 100 views, and the resumed CLI correctly exits 2 with needs_review. Actual property capture and independent measured references remain needed. Evidence: artifacts/research/mac-reconstruction-results.md. FixAnything review: artifacts/research/fixanything-assessment.md. Draft PR: https://github.com/zeeshanhaque21/lingbot-map/pull/2. Fidelity issue: https://github.com/zeeshanhaque21/lingbot-map/issues/3.
## Queued
## Done
