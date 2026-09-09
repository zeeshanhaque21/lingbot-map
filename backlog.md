# Backlog

## In flight
- [ ] mac-property-reconstruction - Implement and validate faithful Mac video-to-mesh reconstruction (since 2026-09-09)
  Native Mac pipeline implemented; full-scene fidelity remains unresolved. Added 32 local comparisons, fresh-context bfloat16/float32 runs, three conservative visibility-carving arms, and camera-pair fitting on reserved feature tracks. Carving fixes none of the seven failing views. Joint rigid fitting improves frame 755 feature error from 5.81 to 1.34 processed pixels on ten reserved matches, but a complete-scene pose optimization is still required. High local depth support can coexist with incorrect native camera motion. Evidence: artifacts/research/mac-reconstruction-results.md. Draft PR: https://github.com/zeeshanhaque21/lingbot-map/pull/2. Fidelity issue: https://github.com/zeeshanhaque21/lingbot-map/issues/3.
## Queued
## Done
