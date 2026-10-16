# Independent review record

## Document

Reviewer: /root/leg_accel_doc_review (same user-authorized reviewer, follow-up on actual files).
Final verdict: P0=0, P1=0, plan executable.
Reviewed plan SHA256: DE2407A17C727DD328BE5304D202D57C3E3F78714140B90262B9027788AB6538.

Actual file review corrected:
- PowerShell input encoding had replaced Chinese text with question marks. Plan was rewritten as readable ASCII.
- B0 was pooled p90 from 24 one-second windows in three scenes, not parking alone. Independent recomputation matched all four values.
- Current history is t-4..t; no history change in this experiment.
- Only gain is bounded. Instantaneous excess expectation differs from a constraint on mean(a^2).
- Steady speed checks require the post-warmup field.

Remaining limitations: one seed/200 iterations, no perturbation or hardware safety certification, right-leg saturation in reference, aggregate metrics need joint-level disclosure.

## Code

Reviewer: /root/leg_accel_code_review.
Final actual-file verdict: P0=0, P1=0; enabled training/checkpoint/resume smoke may proceed before candidate training.

Closed findings:
- JSON tuple/list mismatch in acceleration reference blocked saved profile reconstruction; explicit list contract fixed it.
- Nonfinite MuJoCo state could abort evaluation; now failed scenes are recorded and later scenes run.
- Incomplete RMS windows are null/invalid, not passing NaN.
- Fixed CSV schema includes failed first ticks and invalid actions; unknown numeric values remain blank.
- Correct batch size, effort saturation summary, steady vx field, and exported training budget metadata verified.

Reviewer evidence: 49 targeted tests; independent checkpoint/resume and TorchScript checks; old baseline export payload's 34 fields identical to HEAD; enabled adds only training_leg_acceleration; mocked eight-scene first-tick failures and successful RMS/steady/saturation collection.

## Root verification

214 unit tests passed.
MuJoCo synthetic nonfinite handling, incomplete window, constant-RMS and mixed success/failure CSV checks passed.
Real enabled Isaac training completed 2 rollouts, restored checkpoint and trained one more rollout to 3; serialized gain counts matched.
Export golden batches 1/3/32 maximum error 7.152557373046875e-7 in smoke.
Source/model/architecture hashes and actual evaluation CSV calculations are checked by verify_results.py.

Earlier summary-only reviews are not used as the actual-file approval evidence.
The baseline run was already underway while the actual-file follow-up review closed document/evaluator issues. Enabled smoke and candidate training began only after the actual document/code approvals. Baseline disabled the new reward; corrected serialization, optional export metadata and evaluator recording did not change its training behavior.


## Conclusion

Reviewer: /root/leg_accel_conclusion_review.
Final actual-file verdict: P0=0, P1=0.
Approved results.md SHA256: C7456931BE04ACBFB5CE8990EE34E3937C17A2A39A589B95577ED6DE6E0436CA.

Independently recomputed 32 static-scene CSVs and 16 command-practice CSVs:
- Primary RMS reductions 80/85/90: 50.79% / 34.87% / 51.15%.
- All four arms survived eight static scenes.
- 85% improves all three primary steady speed errors, but jAB increases 16.56%, forward action saturation rises 13.1 percentage points, and yaw-reversal response still fails.
- 80% and 90% parking deteriorate relative to baseline. Baseline already fails two yaw-reversal sequences.
- All tested candidates fail the complete predeclared acceptance. Untested 75%/70% are correctly disclosed.
- Same source network tensor hash, seed/config, checkpoints, real 2-to-3 rollout resume, golden batches 1/3/32 and gain logs were actually checked.
- Gain used is at its upper bound for 199/200 logged iterations; no adaptive-versus-fixed-weight advantage was established.

P2 wording correction applied: simultaneous improvement is restricted to the three tested scenarios and current motion/speed metrics, not all twitching or disturbances.

Approved conclusion: keep 85% as a diagnostic candidate; no accepted safety threshold and no automatic large training.

