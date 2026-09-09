# FixAnything horizontal pilot, 2026-09-09

The complete one-station run generated all 61 images and stitched successfully, but fails the property-fidelity acceptance check.
At input yaw 90 degrees, the rendered passage and blue wall become a frosted partition in the generated image.
The assembled panorama has heavy overlapping ghosts.
No additional station was processed and this candidate was not attached to the property tour.

## Actual outputs

- [Interactive comparison, all frames and video](http://127.0.0.1:8083/fixanything-horizontal-v1/review-v2.html)
- Local generated PNGs: `reconstructions/tours/loop-v1/fixanything-horizontal-v1/frames/000.png` through `060.png`.
- Local generated video: `reconstructions/tours/loop-v1/fixanything-horizontal-v1/generated.mp4`.
- Local stitch: `reconstructions/tours/loop-v1/fixanything-horizontal-v1/stitch/panorama.png`.
- [Measured run receipt and all generated-frame hashes](evidence/fixanything-horizontal-result.json)
- [Generated contact sheet](evidence/fixanything-horizontal-generated.jpg)
- [Input/output cardinal comparisons](evidence/fixanything-horizontal-cardinals.jpg)
- [Stitched candidate](evidence/fixanything-horizontal-stitch.png)

## Reproducible setup

The source was the new horizontal station-000 sweep, rotated around estimated scene vertical with fixed camera center and approximately 19.90 degrees of downward pitch.
Every original input view contained some rendered surface; minimum observed coverage was 18.37% before resizing.
Frames 000 and 060 use the same captured photograph, giving one unique clean anchor.
The other 59 frames use the existing mesh colors.

The pilot retains all 61 frames and uses 512 by 288 pixels, five denoising steps and seed 1 on MPS.
This is one lower-resolution pilot, not a measurement of the released higher-resolution model's best achievable quality.
The released resize operation scales each 832 by 480 input to 512 by 295 with bilinear interpolation, then crops three pixels from the top and four from the bottom.
The copied sweep manifest adjusts intrinsics for that exact pixel-centered transformation and preserves all 61 camera poses and rotations.
Five camera-space points verify the projection transformation to within 5.7e-14 pixels; this checks resize bookkeeping, not reconstruction accuracy.
The original per-frame observed-surface fractions are retained with an explicit note that they were not remeasured after resizing.

```sh
.venv-fixanything/bin/python -u artifacts/research/experiments/run_fixanything_mac.py \
  --sweep reconstructions/tours/loop-v1/sweep-000-horizontal-pilot-v1 \
  --output reconstructions/tours/loop-v1/fixanything-horizontal-v1 \
  --width 512 --height 288 --frames 61 --steps 5
```

Choose a fresh output directory for any rerun; the runner preserves existing results.
The run completed in 606.15 seconds, saved five latent checkpoints, 61 RGB PNGs and both generated and side-by-side MP4s.
The completion record correctly labels `complete_sweep: true`, `coverage_target: horizontal_ring` and `full_spherical_sweep: false`.

## Paired results

Both stitches use the same resized intrinsics, input rotations, 2048 by 1024 canvas and center weighting.
The baseline is the unchanged conditioning sequence, including its source-photo endpoints.

| Diagnostic | Unchanged input | FixAnything output |
| --- | ---: | ---: |
| Mean weighted overlap RGB standard deviation | 0.007526 | 0.088912 |
| Wrap-boundary RGB difference | 0.000710 | 0.000819 |
| Mean per-photo RGB L1 on shared surfaces | 0.066223 | 0.133742 |
| Pixel-weighted RGB L1 on shared surfaces | 0.060321 | 0.118177 |

The overlap diagnostic worsens by 11.81 times.
Its baseline includes black unobserved regions, so it is not an independent architectural metric; visible ghosting and changed passage appearance provide additional evidence.
The low wrap-boundary number does not contradict failure inside the panorama, and the two identical source anchors give little independent closure evidence.
The first generated anchor differs from its input by mean RGB L1 0.009764; the final anchor differs by 0.023941.
Clean-frame conditioning does not enforce identical output pixels.

The source comparison uses all 24 non-anchor photographs excluded from mesh fusion, with 1,053,257 shared visible pixels.
Every photograph's color error worsens.
Candidate and baseline use identical mesh-visibility and horizontal-direction masks; unsupported directions are excluded from both.
The do-nothing baseline compared against itself produces identical scores.
These photographs still contributed to camera estimation and reconstruction context, so the result is a diagnostic rather than independent geometry ground truth.
The reference comparisons for frames 5, 165 and 215 were also inspected visually.

All 61 generated images were inspected in a contact sheet, with larger cardinal pairs and the stitched panorama inspected separately.
The output changes direction but does not follow the supplied yaw geometry consistently.
Frames around 90 degrees retain a partition where the conditioning image shows a passage, while the rear sweep becomes smeared reflective surfaces.
Later views introduce detailed shelving that is unsupported by the supplied mesh pixels; this experiment does not establish whether those details correspond to the actual building.

The horizontal sweep covers 21.49% of the equirectangular canvas under this camera calibration.
This fraction counts canvas pixels rather than equal-area solid angle, and measures camera directions rather than observed building surfaces.
The curved band reflects the pitched reference coordinate system; its black poles are outside the requested horizontal pass.
A complete horizontal turn is therefore not a complete spherical panorama.

## Decision and next diagnostic

Reject this candidate for the property tour.
Keep the corrected horizontal camera export, exact resized calibration and preserved source anchors.
Passing the nonempty-input check is necessary but does not establish that FixAnything will preserve camera direction or architecture.

The next bounded diagnostic should run one published example through the same MPS adapter and compare it with the published behavior before another property-generation attempt.
If that control works, test stronger distinct source-view support on one station with a do-nothing baseline.
The current evidence does not separate sparse conditioning, the single unique anchor, pilot resolution/step count and MPS numerical behavior as possible contributors.
Increasing resolution, steps or station count without that control would not identify the cause.
