# Method

## Task

A simulated SMPL humanoid follows a HUMANISE reference in a ScanNet room. A short pelvis push disturbs the motion. Success requires reaching the **original task goal**, standing stably, and avoiding contact between **any body part** and a specified forbidden object.

The task combines balance recovery, reference tracking, goal completion, and object-specific contact constraints. It does not evaluate getting up from arbitrary fallen poses or transfer to a physical robot.

## Language and task binding

`scripts_demo/vlm_relations.py` uses Qwen3-VL-4B-Instruct to parse a restricted instruction grammar. Known scene instances and metric geometry resolve the parsed `nearest` / `farthest` relations. The language model is not asked to predict physical forces or object strength.

`scripts_surface_demo/run_system.py` implements the audited office-chair end-to-end example. It deliberately checks the exact task, source `015317`, goal instance `10`, forbidden instance `17`, reference identity, and model identity. It is **not a general-purpose arbitrary-instruction API**.

The Qwen process runs before physics. Language inference is not part of the 30 Hz control loop.

## Closed-loop correction

```text
Current state + contacts + reference + task geometry
              + four-frame history
                         ↓
                  H1 BC correction
                         +
            frozen same-state terrain prior
                         ↓
                  69D joint action
                         ↓
           physics step and contact feedback
```

The student target is:

```text
correction = teacher_executed_total - same_state_composed_terrain_prior
```

Actions are composed in normalized, tanh-transformed actuator space. The physics interface applies the inverse transform and the existing PD control path. The prior is recomputed at every control step; H1 predicts only the current correction. Future push direction and timing are not observations.

Relevant files:

- `scripts_student_surface/model.py`: BC and diffusion networks.
- `scripts_student_surface/controller.py`: inference and prior/correction composition.
- `scripts_teacher_planned/warm_prior.py`: frozen terrain prior.
- `scripts_teacher_planned/recovery_runtime.py`: executed action contract.
- `scripts_teacher_planned/warm_task.py`: task features and correction behavior.
- `scripts_teacher_planned/v5_task.py`: physical evaluation and termination.

## Model and training

| Item | Selected configuration |
|---|---|
| Policy | H1 BC |
| Weights | EMA |
| Condition encoder | `obs_dim → 256 → 256`, SiLU |
| Output head | `256 → 69` |
| Prediction / execution horizon | 1 / 1 |
| History | 4 observation frames |
| Data | 2,609 successful trajectories; 174,390 supervised frames |
| Updates / batch size | 15,000 / 256 |
| Training seed | 109 |
| EMA | Maximum decay 0.9999; power 0.75 |
| Data streams | Teacher full / takeover / late takeover; BC and diffusion early / hold takeover |

The observation includes physical state, contact, reference, goal and forbidden geometry, terrain samples, and foot-clearance features. Sampling balances streams and task families, then available force/direction/episode/frame strata. Force-stratum target mass is 20% / 40% / 40%; unavailable successful strata are renormalized rather than fabricated.

The diffusion control uses a temporal network, 100-step cosine schedule, velocity prediction, and eight DDIM inference steps. BC has no iterative denoising.

## Reference and geometry fixes

1. Align character placement, scene ground queries, and reference height.
2. Allow bounded reference changes near forbidden geometry while preserving the original numerical goal.
3. Represent foot clearance using collision geometry rather than only approximate body points.
4. Collect teacher corrections from states actually reached by the student.
5. Align prediction with execution using H1.

Historical H6, aggregation, first-step-weighting, and H1 runs changed more than one setting across the full sequence. Their progression is engineering history, not a clean single-factor causal ablation.

## Success and termination

For the room-based task, the main conditions are:

- Fall: root height below 0.50 m, root up component below 0.35, or non-foot ground contact above 20 N after the initial guard.
- Forbidden contact: designated object contact force above 2 N; the violation flag is accumulated.
- Stable goal: root XY distance below 0.35 m, height above 0.65 m, root up above 0.85, target polygon distance at most 0.8 m, XY speed at most 0.5 m/s, absolute vertical speed at most 0.5 m/s, and torso tilt at most 45°.
- Completion: 30 consecutive eligible stable frames, with no accumulated fall or forbidden-contact failure.

Eligibility also requires the reference-motion time and post-push observation window to be satisfied. The separate recovery-time measure uses a shorter balance window; it is not equivalent to task success.

## Video interpretation

The original Mesh is a kinematic reference. The two capsule columns are recorded physical executions. Rendered body positions are never replaced with generated motion.

All English videos use the same presentation timeline: a start hold, every source frame at 0.5× speed, and a final hold. The baseline may terminate before H1 or the reference; the frozen panel is labeled explicitly. Goal and forbidden markers are projected using the original camera. The lounge contact alert uses an exact-pose sensor replay that identifies `L_Hand`.
