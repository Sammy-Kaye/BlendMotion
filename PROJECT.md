# BlendMotion

Working codename in docs: Handle.

Goal: make posing and animating in Blender dramatically easier. Drag a hand, foot, hips or head and the whole body follows naturally. Then edit motion with the same direct-manipulation feel.

Reference product: Animatica Marionette (closed source, $59, Blender 5.0+, CPU pose model, FK key output). We are building our own from the ground up. Nothing of theirs is available, so nothing is copied.

## Principles
1. Ship something usable at every phase. No big-bang ML bet.
2. Classical solver first, learned model second, behind the same interface.
3. Output is plain FK keyframes on the user's own rig. No rig layer, undo works.
4. Any humanoid rig works without renaming bones (auto-mapping + manual override).
5. Everything testable headless (Blender CLI) and visually (Blender MCP screenshots).

## Architecture
- `handle/rig`: bone-name mapping to a canonical skeleton, T/A-pose normalization, read/write FK.
- `handle/solver`: `Solver.solve(pose, targets) -> pose`. Implementations: `FullBodyIK` (classical, phase 1), `LearnedPrior` (ONNX, phase 2).
- `handle/ui`: viewport handles (GPU draw + modal operator), panel, hand presets.
- `handle/motion`: reach (edit propagation), trails, smoothing, foot-lock, motion check. Mostly signal processing, no ML.
- `training/`: dataset prep, model, training, ONNX export. Runs on cloud GPU.

## Roadmap
| Phase | Outcome | Needs ML | Cloud credits |
|---|---|---|---|
| 0 | Repo, add-on skeleton, rig mapper, test rigs, headless test harness | no | no |
| 1 | Drag hand/foot/hips/head, full-body IK, foot planting, FK key output | no | no |
| 2 | Learned pose prior (natural-looking whole-body response), ONNX on CPU | yes | yes, main use |
| 3 | Motion trails, reach, zoetrope, smooth brush, foot lock, motion check, finger presets | no | no |
| 4 | Extras that go beyond Marionette: pose from reference image, text to pose, pose library | yes | yes |

## ML plan (phase 2)
- Task: given partial targets (end-effector positions/rotations) + current pose, predict full joint rotations. Prior art to study: ProtoRes, SMPL-IK (boreshkinai), VPoser, SparsePoser.
- Model: small MLP/transformer, target under 20 MB and under 15 ms on CPU. Export ONNX, run with onnxruntime inside Blender.
- Data: BLOCKER, depends on commercial vs personal use. AMASS forbids training for commercial use. Permissive alternatives must be license-checked individually (CMU raw, LAFAN1, 100STYLE, own mocap/Mixamo clips).
- Training: the model is small. Local RTX 3050 (4 GB) can train a first version; free Kaggle/Colab GPUs for bigger runs. The client's Claude credits (expire 2026-11-04) fund agent work, not GPUs, so research, data prep and code are front-loaded before that date.

## Agent team (definitions in `.claude/agents/`)
- `architect`: owns design, interfaces, roadmap, reviews.
- `blender-dev`: add-on code, operators, GPU drawing.
- `ml-trainer`: datasets, model, training, export.
- `blender-qa`: runs Blender (CLI + MCP), screenshots, regression tests on real rigs.

## Open decisions (client)
1. DECIDED: personal / open source. AMASS and other research datasets allowed (keep non-commercial notice in README).
2. DECIDED: credits are Claude credits, expire 2026-11-04. Work is front-loaded and parallelised across agents.
3. DECIDED: repo is github.com/Sammy-Kaye/BlendMotion.
4. A rig you actually animate with.

