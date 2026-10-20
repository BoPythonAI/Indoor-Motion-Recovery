# Results and evaluation scope

The selected model is **H1 BC / EMA**, checkpoint SHA256:

```text
b3098abe5d13fda49c29d785f566f9fa5c96e332231ec105b8e33bc70a0ac4e2
```

Frozen selection status: `FROZEN_QUALIFIED`. Statistics below come from the saved physical evaluation records in [`evidence/`](../evidence/). Public JSON copies rewrite machine-specific path prefixes; the original file hashes are retained in `report_provenance.json`.

## Fixed protocol

Each source has 24 slots: three strengths (`0`, `0.45`, `0.75 BW`), four world directions, and two reference phases (`0.3`, `0.6`). Pushes act on the pelvis for six 30 Hz control steps, or 0.20 s.

Development sources: `011723`, `015317`, `015542`, `015547`, `015571`. Each strength has 40 recorded cases. Direction labels at zero force do not create independent physical perturbations. Sources were reused during development.

| Method | No push | 0.45 BW | 0.75 BW |
|---|---:|---:|---:|
| Same-reference baseline | 39/40 (97.5%) | 33/40 (82.5%) | 21/40 (52.5%) |
| Scene-correction teacher | 38/40 (95.0%) | 32/40 (80.0%) | 30/40 (75.0%) |
| **H1 BC** | **38/40 (95.0%)** | **33/40 (82.5%)** | **27/40 (67.5%)** |
| H1 diffusion | 34/40 (85.0%) | 33/40 (82.5%) | 29/40 (72.5%) |
| Nominal-only H1 BC | 37/40 (92.5%) | 35/40 (87.5%) | 30/40 (75.0%) |

Original development gates: **38 / 32 / 24** successes. H1 BC passes all three. Diffusion and nominal-only BC do not pass the no-push gate, despite higher strong-push scores. The nominal-only control uses 80 trajectories / 8,622 frames versus 2,609 mixed-recovery trajectories; it is not a data-volume-matched causal ablation.

## Baseline-to-BC paired transitions

![Observed success-rate change for H1 BC relative to baseline.](assets/charts/baseline_gain.svg)

| Outcome transition | No push | Medium | Strong |
|---|---:|---:|---:|
| Failure → success | 1 | 3 | 7 |
| Success → failure | 2 | 3 | 1 |
| Both succeed | 37 | 30 | 20 |
| Both fail | 0 | 4 | 12 |

The strong-push gain is six net successes, or **15 percentage points**. No-push performance is 2.5 points below the baseline; medium-push success is unchanged.

## Failure flags

| Strength | Baseline falls / forbidden contact | H1 BC falls / forbidden contact |
|---|---:|---:|
| No push | 0 / 1 | 2 / 0 |
| Medium | 3 / 4 | 0 / 7 |
| Strong | 13 / 6 | 5 / 8 |

Flags can overlap and are not guaranteed to be mutually exclusive counts. Missing timeout fields are inferred only for otherwise unclassified terminal failures. BC reduces strong-push falls, while forbidden contact remains a major limitation.

## Per-source success

![Strong-push success counts by development source.](assets/charts/source_breakdown.svg)

Entries are no / medium / strong successes; each number has denominator 8.

| Source | Baseline | H1 BC | H1 diffusion |
|---|---|---|---|
| 011723 | 8 / 6 / 1 | 6 / 7 / 4 | 3 / 7 / 5 |
| 015317 | 8 / 6 / 5 | 8 / 6 / 5 | 8 / 6 / 6 |
| 015542 | 8 / 8 / 7 | 8 / 8 / 8 | 8 / 8 / 8 |
| 015547 | 8 / 8 / 5 | 8 / 8 / 6 | 8 / 8 / 7 |
| 015571 | 7 / 5 / 3 | 8 / 4 / 4 | 7 / 4 / 3 |

## Validation and reserved motion

Validation sources `014810` and `015330`: **16/16 at every strength**. Gates were 16 / 13 / 10. Two saturated sources do not demonstrate universal robustness.

Reserved source `011387`, an unseen motion in a **known room**:

| Method | No push | Medium | Strong |
|---|---:|---:|---:|
| Baseline | 8/8 | 6/8 | 4/8 |
| H1 BC | 8/8 | 7/8 | 7/8 |

This source was evaluated after the model was frozen and is not added to the development denominator. It is not an unseen-room benchmark.

## Fresh language-to-physics execution

A new Qwen parse for source `015317` resolved goal 10 and forbidden object 17. The subsequent 24-case physical run scored **8 / 6 / 5**, matching the source-level result.

The recorded strong-push pair uses slot 22, direction 3 and phase 0.3. Baseline falls at step 41; H1 completes 100 steps with 30 stable frames, approximately 0.00484 m goal distance and no forbidden contact. `matched_pair.json` reports `matched=true`; the repair recording is the actual fresh-run trace.

## Three published demonstrations

| Scene | Source | Baseline / H1 physical frames | Baseline failure | H1 outcome |
|---|---|---:|---|---|
| Office chair | 015317 | 41 / 100 | Fall; table contact 0 N | Stable goal, no forbidden contact |
| Bedroom door | 018868 | 42 / 109 | Fall; bed contact 0 N | Stable goal, no forbidden contact |
| Lounge armchair | 017761 | 52 / 106 | Left-hand contact, 36.7175 N | Stable goal, no forbidden contact |

All are selected failure-to-success examples. The original project contains eight gallery scenes; this public showcase includes the three requested demos.

## Latency

Saved measurements are for a **batch of 24 environments**, excluding physics, contact processing, IO and Qwen parsing.

| Source | Student + priors p50 / p95 (ms) | With task flat inference p50 / p95 (ms) |
|---|---|---|
| 011723 | 1.657 / 1.705 | 2.514 / 2.588 |
| 015317 | 1.675 / 1.733 | 2.541 / 2.610 |
| 015542 | 1.656 / 1.694 | 2.514 / 2.555 |
| 015547 | 1.664 / 1.853 | 2.520 / 2.733 |
| 015571 | 1.499 / 1.551 | 2.323 / 2.401 |
| 014810 | 1.558 / 1.599 | 2.410 / 2.471 |
| 015330 | 1.512 / 1.565 | 2.342 / 2.400 |

Do not divide these timings by 24 to report a single-character end-to-end latency, or average per-source percentiles into a pooled percentile.

## Completion and next steps

The defined engineering scope is complete: frozen controller, development and validation gates, reserved-motion evaluation, language-to-physics example, matched recording and annotated gallery. The release verifies file integrity and presentation; it does not rerun GPU simulation.

Further research should prioritize forbidden-contact failures, no-push regressions, independent unseen-room evaluation, multiple training seeds, and data-volume-matched comparisons. Broad free-form language support, physical-robot transfer and paper submission are outside the completed scope.

## Reproduce the figures

![Five-controller comparison across the three push strengths.](assets/charts/controller_comparison.svg)

All figures are generated from the frozen JSON files in [`evidence/`](../evidence/), with the same counts used in the tables above. Success-rate axes start at zero; the change plot uses a signed axis centered on a labeled zero line. Counts and denominators are displayed explicitly.

```bash
python -m pip install -r requirements-plots.txt
python tools/plot_results.py
```

Outputs: three SVG figures, matching PNG exports, and [results.csv](assets/charts/results.csv). The [plotting script](../tools/plot_results.py) reads all five controller summaries directly. No training or physics rerun is needed to regenerate the plots.

The figures report recorded proportions. Error bars are omitted because the evaluation does not provide independent repeated training seeds, and zero-force direction slots do not constitute independent perturbations. These figures do not establish statistical significance.
