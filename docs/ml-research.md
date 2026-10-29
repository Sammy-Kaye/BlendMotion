# ML research: learned pose model for Handle (phase 2)

Date: 2026-10-29. Scope: personal / open-source, non-commercial.
Evidence labels used below: [ran] executed here, [read] read in repo/paper text, [recall] from memory, not re-checked.

## 0. Decisions in one screen

1. Model: "ProtoRes-lite", about 4.1M params, residual on the current pose, 6 fixed effector slots with mask. Written from scratch, no third-party code copied.
2. Skeleton: canonical 22-joint SMPL body (joints 0-21), rot6d outputs, rig layer retargets to and from it.
3. Accuracy: do not expect <1 cm from the network alone. Ship network + analytic two-bone limb correction + optional few-step IK polish in the solver. Whether <1 cm is reached is UNMEASURED until trained.
4. Data: AMASS (non-commercial, ok for this project) + CMU raw (via AMASS) + LAFAN1 optional (NC-ND, see caveat). Do not redistribute data. Weights: see licensing caveat in section 3.
5. Training: local RTX 3050 4 GB is sufficient for a 4M-param model; Kaggle/Colab for sweeps.
6. Latency: 0.5 ms (ORT, 1 thread, random weights) [ran]. 30x headroom under 15 ms.

## 1. Reference repos (what is actually in `references/`)

| Item | License found | Role |
|---|---|---|
| `protores/` | NO LICENSE FILE; repo is only README + gif ("transferred to Unity Labs repo") | paper only |
| `smpl-ik/` | `LICENSE.md`: "License for non-commercial academic research purposes only smpl-ik copyright (c) 2022 Unity Technologies ApS" | ProtoRes-style code + AMASS pipeline |
| `human_body_prior/` | `LICENSE`: "Software Copyright License for non-commercial scientific research purposes" (Max Planck, MPI) | VPoser, SMPL-X tooling |
| `BVH-Motion-Retargeter/` | `LICENSE`: MIT License (c) 2026 Junius Tang | Blender BVH retarget add-on |
| `Humanoid-Retarget-Godot/` | No LICENSE file; README says "Licensed under the MIT License" (D3ZAX) | Blender humanoid retarget script |
| `Keemap-GPL3/` | `LICENSE`: GNU General Public License v3 | Blender bone-mapping add-on |
| SparsePoser | No repo supplied; paper only (ACM TOG, Ponton et al. 2023) | paper only |

### protores (paper `protores_paper.pdf` [read])
- Does: maps a variable set of effectors (position, rotation, look-at, each with tolerance) to full-body pose. Effector tokens (joint-ID and type embeddings + 6D data + tolerance) pass through a "proto-residual" encoder: N-BEATS-like FC blocks, each block's forecast is added to the pose embedding, the pooled (mean) embedding is subtracted from the next block's backcast. A decoder draft-predicts 3D joint positions (GPD) then predicts rotations (IKD) conditioned on them. Loss: weighted L2 on positions + geodesic rotation loss, with per-effector random tolerance/noise.
- Numbers from the paper (Table 1): ProtoRes 41M params, 5.5 ms per solve on RTX 2080 Super, beats FinalIK and Transformer on reconstruction. Width 1024, 3 blocks x 3 layers, batch 2048, up to 40k epochs.
- Reuse: ideas only (set-pooled encoder, tolerance-as-input, noisy-effector training, positional draft then rotation head). Nothing to copy; license unknown, and paper is not code.
- Our departure: 41M params is 164 MB fp32, over our 20 MB budget. We use ~10x smaller width.

### smpl-ik (`references/smpl-ik` [read])
- Does: ProtoRes variant (`smplik/modules/protores.py`, `WeightedProtoRes`) conditioned on SMPL shape (morphology-aware), trained on AMASS, with an inversion model. Config `backbone_default.yaml`: width 1024, 3 blocks x 3 layers per stage, embedding_dim 32.
- License is restrictive: "Reference Purposes" = non-commercial academic research only; derivative works must carry the same license and attribute it. Even for a personal MIT/Apache open repo, copying its code would force the copied files (and works derived from them) under that license, and our repo would not be "OSS" by OSI terms.
- Policy: read for ideas (weighted encoder, losses in `smplik/losses/`), do NOT copy code. Requirements are pinned to torch 1.9 + pytorch3d, which is another reason not to use it.

### human_body_prior / VPoser (`references/human_body_prior` [read])
- Does: VPoser, a VAE pose prior on SMPL body pose in a latent space; also AMASS-prep tutorials.
- License: MPI non-commercial scientific research license. It explicitly "prohibits the use of the Model & Software to train methods/algorithms/neural networks/etc. for commercial use of any kind", forbids redistribution ("No Distribution": not copied, shared, distributed) and forbids making it available in modified form to third parties without permission. So do not vendor any of its code into our public repo; do not ship VPoser weights.
- Ideas to reuse: a latent pose prior / pose-likelihood regulariser. We may use VPoser later (Phase 4, pose from image / pose library) as a user-installed optional download, never bundled.

### BVH-Motion-Retargeter (MIT [read])
- Does: BVH to Mixamo/UE5/VRM retargeting in Blender (constraint-based, bake, cleanup).
- Reuse: MIT allows copying with the copyright notice kept in copied files. Candidate for its rig-name mapping tables and test cases, only if rig layer needs them. We have not audited the code quality; read before copying.

### Humanoid-Retarget-Godot (MIT per README [read])
- Does: one Blender script mapping humanoid bones for Godot. No LICENSE file, so legal footing is only the README sentence. Use as idea source for bone-name tables; if code is copied, include the author's name and note the missing license file.

### Keemap-GPL3 (GPL v3 [read])
- Does: Blender rig-to-rig motion retarget with user bone mapping saved to files.
- GPL v3 is copyleft: copying code into the add-on makes the add-on GPL v3 (Blender add-ons are generally GPL anyway). Decide the repo license before copying. Safest: read for UX ideas (mapping list, save/load of mapping files), write our own. Not usable in the training code at all.

### SparsePoser (`sparseposer_paper.pdf` [read])
- Does: 6-point VR full-body from sparse trackers. Convolutional autoencoder generates a continuous motion (dual quaternions), then per-limb feed-forward "learned IK" networks correct arms and legs. Loss L_S = MSE of FK(end-effectors) between ground truth and corrected pose; L_Reg = MSE keeping non-effector joints close to the generator. lambda = 0.1, AdamW, lr 1e-4, batch 256, own mocap ~1M poses (4.5 h at 60 fps, mirrored to 2M). Its own finding: the generator alone is imprecise at end-effectors, hence the separate learned IK stage.
- Reuse: the architectural lesson (global net for naturalness + local limb IK for exactness) and the FK-space loss. It targets VR time series; ours is single-pose edit, so the conv autoencoder does not apply.

## 2. Model design

### 2.1 Problem framing
Drag-to-pose is an under-determined single-frame IK problem: effectors set = subset of {hips, head, L/R hand, L/R foot}, plus (optionally) rotations; the net must keep unconstrained joints near the CURRENT pose (so dragging a hand does not change the legs unpredictably) and make the rest plausible.

### 2.2 Comparison

| | ProtoRes | SMPL-IK | SparsePoser | Ours (ProtoRes-lite) |
|---|---|---|---|---|
| Input | variable effector set, tokens + pooled | same + SMPL shape | 6 trackers over time window | 6 fixed slots with mask + current pose |
| Params [read] | 41M | similar scale (cfg width 1024) | not stated in text read | 4.1M [ran] |
| Output | rotations (+ draft positions) | rotations | dual quats per frame | residual rot6d delta + root pos |
| Handles absent effectors | set pooling | set pooling | no (fixed 6) | mask + mean pool |
| Precision fix | none (relies on loss) | none | learned limb IK | analytic two-bone IK post-step |
| Conditioning on current pose | no | no | window of previous frames | yes (core to drag UX) |
| Temporal | no | no | yes | no (frame-independent, residual) |

Why fixed slots instead of a generic token set: the add-on only exposes 6 handles, so slot embeddings are enough and the graph is static for ONNX/ORT. A generic joint-ID token set (ProtoRes) could be added later for elbows/knees "pole" hints by growing the slot count (slots are 11-dim, cheap).

### 2.3 Canonical skeleton and arbitrary rigs
- Canonical: SMPL joints 0-21 (22 joints; `training/skeleton.py`). Y-up, +Z forward, left = +X, T-pose rest with identity rotations.
- Rig layer (add-on side, not in training) does: (1) bone mapping (name aliases + manual override) to the 22 joints, with spine count != 3 handled by distributing rotation across the spine chain; (2) compute for each mapped bone the rest-frame rotation offset between the user's bone rest orientation and canonical T-pose orientation, so a user local rotation = R_rest^-1 * R_canon * R_rest style conjugation; (3) uniform scale by pelvis-to-head or leg+arm length to a 1.70 m figure; (4) after inference, convert back and optionally apply per-limb length mismatch correction via analytic two-bone IK so hands actually reach the dragged targets on non-SMPL proportions.
- Proportions: SMPL has real shape variety. Train with bone-length augmentation (+-12% per limb group) with FK using per-sample offsets (`SkeletonFK.forward(offsets=...)`) and feed 4 numbers (leg, arm, torso, neck+head scale ratios) to the net as extra pose-encoder input. That is a small change to `POSE_DIM`; not in the stub.
- Missing canonical joints (e.g. no collars): fix the collar rotation to identity in the output.

### 2.4 Concrete layers (as implemented in `training/model.py`)
- Effector token (11) = [mask, pos xyz, rot6d, tolerance]; plus learned slot embedding (16) -> Linear(27->256) -> ReLU.
- Encoder: 2 ResBlocks (N-BEATS style: 2x Linear(256,256)+ReLU, backcast Linear(256,256), forecast Linear(256,256)); forecasts summed; between blocks subtract masked-mean embedding / (i+1) from the token stream (ProtoRes proto-residual). Final masked mean -> 256-d.
- Pose encoder: Linear(135,256)-ReLU-Linear(256,256), where 135 = 22*6 rot6d + pelvis position relative to anchor.
- Decoder: Linear(512->512) then 3 ResBlocks (width 512, forecast 512) summed, ReLU, Linear(512->135); head zero-initialised so untrained model is identity.
- Output = cur_pose + delta; rot6d orthonormalised by Gram-Schmidt downstream (in FK, and in add-on numpy).
- Inputs expressed relative to the anchor (centroid of active positional effectors), as ProtoRes does for translation invariance. Heading: rotate everything so the pelvis forward is +Z (yaw-canonicalised) on the add-on side and inverse after.

### 2.5 Parameters and latency (justified)
- Count [ran]: 4,117,991 params = 16.5 MB fp32, 8.2 MB fp16. Dominated by decoder: 3 blocks x ~1.05M (2x512^2 + 2x512^2 for back/fore) ~ 3.1M.
- Exported ONNX (opset 17, fp32) size: 15.17 MB [ran]. Under 20 MB, but with little headroom if we widen. Option: fp16 weights (about 8 MB) or dec_width 384 (about 2.4M params).
- MACs ~ params for batch 1, so about 4M MACs. At 1 thread this is memory-bound (16 MB of weights), which matches measurement.
- Measured on this machine (Windows 11, 6-thread torch / ORT, random weights, batch 1, 500 runs) [ran]: ORT 1 thread mean 0.52 ms, p99 0.81 ms; ORT 4 threads mean 0.35 ms; torch CPU 1.49 ms. Torch vs ORT max abs diff 1.2e-7 with a randomised head [ran]. This excludes the add-on's own overhead (bpy reads/writes of ~22 bones, rig conversion, IK polish), which is likely larger than inference. Target 15 ms is therefore dominated by Python/bpy glue, not the net. Not measured: ORT inside Blender's bundled Python, other CPUs.
- Estimated alternatives (arithmetic, not run): ProtoRes-size (width 1024) would be about 40M params, 160 MB: rejected.

### 2.6 Reaching <1 cm (honest plan)
Papers report pose-level losses, not guaranteed sub-cm. FinalIK/ProtoRes comparisons are in normalised units; I did not find a cm number in the text I read. Plan:
1. Network with FK end-effector loss -> expect a few cm on held-out (guess, unmeasured).
2. Solver post-step: analytic two-bone IK per limb for hand/foot targets (exact if reachable), keeping network's elbow/knee direction as pole vector. Gives ~0 cm for reachable limbs. Head/hips are directly placed, so near 0.
3. So "mean effector error <1 cm" is met mostly by step 2. Report both numbers separately (raw net and net+IK) so the claim stays honest. The "naturalness" of the rest is the network's contribution.

## 3. Data plan

### 3.1 Licenses (public non-commercial repo)
| Dataset | License (source) | Implication |
|---|---|---|
| AMASS | MPI non-commercial "scientific research, education, or non-commercial artistic projects"; no commercial training; no redistribution of data [web search of amass.is.tue.mpg.de/license.html, summary only] | Allowed for this project. Never commit AMASS data or derived arrays to the repo. Keep a NON-COMMERCIAL notice. |
| AMASS sub-datasets (CMU, KIT, BMLmovi, HDM05, Eyes_Japan, etc.) | each has its own license on top; AMASS page lists them [recall] | Check the per-subset list on the download page before use. |
| SMPL / SMPL-H body models | MPI non-commercial license, separate registration [recall] | Only needed to get neutral rest joints (one small computation). Do not commit model files. |
| CMU Mocap (raw) | "free for all uses" per mocap.cs.cmu.edu [recall] | Most permissive. CMU through AMASS is usable but through AMASS terms. CMU BVH direct is a clean fallback for a commercial future. |
| LAFAN1 (Ubisoft) | CC BY-NC-ND 4.0 [web search, confirmed via GitHub page] | Non-commercial OK. "NoDerivatives": training a model is arguably not a derivative, but conversion to our skeleton and redistribution would be. Use locally only, do not share converted data. 4.6 h, 5 actors, 30 fps. |
| 100STYLE | CC BY 4.0 [recall] | Good for variety. Verify at download. |
| Mixamo clips | Adobe terms: free use in projects, but not redistribution or training of ML for sale [recall; check current terms] | Skip for training. |
| HumanML3D / text-to-motion | derived from AMASS; its authors distribute scripts, not data [recall] | Phase 4 only. |

Weights licensing: weights trained on AMASS inherit the non-commercial restriction ("train ... for commercial use of any kind" is prohibited). Public NC release of weights is a gray area; AMASS forbids redistribution of the Dataset, but weights are not the dataset. Ask Max Planck (ps-licensing@tue.mpg.de) if unsure, or release only the training code and let users train. Also keep a CMU-only variant as a possible clean-licence escape hatch (lower quality).

### 3.2 What to download (user must do; I downloaded nothing)
Needed AMASS format: "SMPL-H G" `.tar.bz2` per sub-dataset (npz files with `poses`, `trans`, `mocap_framerate`, `gender`).
Suggested first pass (rough sizes from memory, check page): CMU (~2 GB), BMLmovi, KIT, HDM05, ACCAD, Eyes_Japan_Dataset, MPI_HDM05, TotalCapture, SFU, Transitions_mocap, DanceDB optional. Full AMASS about 40+ h of motion / ~20 GB compressed [recall]. Hold out whole sub-datasets for generalisation tests (see section 4).

### 3.3 User registration steps (explicit)
1. Create account at https://amass.is.tue.mpg.de, accept the licence, download SMPL-H G archives for the subsets above.
2. Create account at https://smpl.is.tue.mpg.de (and https://mano.is.tue.mpg.de for SMPL+H if required by the AMASS download page), download the SMPL neutral model `.pkl`/`.npz`. Only used to compute rest joint offsets once.
3. Place files at `D:\data\amass\<subset>\...npz` and `D:\data\smpl\SMPL_NEUTRAL.pkl` (outside repo, outside OneDrive).
4. Optional: LAFAN1 from github.com/ubisoft/ubisoft-laforge-animation-dataset (git-lfs archive), CMU BVH from the CMU mocap site.

### 3.4 Preprocessing pipeline (to write, `training/data/`)
1. For each npz: take `poses[:, :66]` (root + 21 body joints axis-angle; drop hands), `trans`, fps.
2. Resample to 30 fps (frame skip for 60/120/100 fps; ignore clips <1 s).
3. Axis-angle -> rotation matrix -> rot6d (`fk.axis_angle_to_matrix`, `matrix_to_rot6d`).
4. AMASS z-up -> our y-up: apply fixed rotation (-90 deg about X) to root orientation and trans. Verify with a sanity plot: feet below pelvis on +Y axis.
5. Compute rest offsets once from SMPL neutral joints (J_regressor @ v_template, T-pose), write into `skeleton.py`, replacing the approximate table. Regression test: FK of a known AMASS frame vs SMPL joint output within 1 mm (requires smplx; one-off).
6. Foot-contact/ground: shift each clip so the min foot height of the first frame sits at 0; drop clips whose feet are far above ground for the whole clip (flying).
7. De-duplicate near-identical frames: keep frames with joint displacement > 1 cm from last kept frame (static poses dominate AMASS). Target ~3-5 M poses.
8. Pack to a single float16 `.npy` memmap: shape (N, 22, 6) + root trans (N, 3) + clip id + subset id. About 3M x 135 x 2 B = 0.8 GB.
9. Splits by SUBSET/subject, not by frame (section 4).
LAFAN1/CMU BVH converted via the same joint mapping (name table) then step 3 onwards; BVH Euler to matrix requires per-file rest-pose handling; these need the retarget step and are riskier than AMASS.

## 4. Training and evaluation plan

### 4.1 Sample generation (online, per batch, on GPU)
1. Draw a pose P (target) from the packed set; draw a "current pose" C: either (a) another pose from the same clip within +-2 s, (b) P with noise on joints, or (c) the rest pose, mix 50/30/20.
2. Choose effector subset: random non-empty subset, biased toward common drags (single hand 25%, single foot 15%, hips 10%, head 10%, two or more 40%). Positions are FK(P), rotations (rot6d) enabled with probability 0.3 per active effector.
3. Tolerance Λ~U[0,1] per effector; add noise scaled by Λ to position/rotation inputs, loss weight 1/sigma(Λ) clipped (ProtoRes idea, [read]).
4. Effector positions anchored to centroid of active ones.

### 4.2 Losses
- L_pos: FK effector position error (L2 in metres, weight ~100), active effectors weighted by 1/noise.
- L_rot: geodesic distance on all joint rotations vs P (this is where naturalness comes from).
- L_fk_all: L2 on all 22 FK joint positions (0.1 weight): stops drift in non-effector joints.
- L_stay: for joints not downstream of an active effector, L2 between output and C rotations, only in samples of type (a)/(b), small weight (0.1): keeps unconstrained parts stable. Needs tuning; this competes with L_rot, so the target-vs-current mix matters. [Design guess, unverified.]
- L_contact (later): penalise foot sliding is not applicable to single-pose.
- Optionally SparsePoser-style L_Reg for post-IK consistency.

### 4.3 Augmentation
- Mirror (swap left/right, negate X and the appropriate rot components via `MIRROR_PERM` + reflection): doubles data.
- Random yaw about the vertical axis and global translation (inputs are anchor-relative so mostly free).
- Bone-length scaling (+-12% limbs, +-8% torso/neck) via per-sample FK offsets; targets recomputed from scaled FK.
- Pose noise on C. Random effector dropout. Tolerance noise as above.

### 4.4 Split and metrics
- Hold out by sub-dataset/subject: e.g. validation = ACCAD + one BMLmovi subject set, test = HDM05 + Eyes_Japan (never touched until final). Frame-level random splits leak near-duplicate frames and would over-report.
- Metrics on test set, reported separately for "raw net" and "net + two-bone IK":
  - mean / median / p95 end-effector position error (cm) for hands, feet, head, hips.
  - mean per-joint position error (MPJPE, cm) over all 22 joints vs the real pose.
  - mean joint angle geodesic error (deg).
  - Stability: change in non-dragged limbs (cm) per dragged effector, vs ground-truth-free baseline of the current pose.
  - Baselines: current pose unchanged; classical FullBodyIK from phase 1 once exists.
  - Per-case tables: 1 effector, 2, 5-point benchmark like ProtoRes.
  - Latency re-run of `benchmark.py` after training (ORT, inside Blender Python).

### 4.5 Time estimate on RTX 3050 4 GB
Not measured (no GPU run here). Reasoning: 4M-param MLP with batch 2048 uses well under 2 GB; the 22-step python FK loop (the sequential chain) is likely the bottleneck, not the net, and roughly 10-30 ms/step is plausible. At 20 ms/step, 200k steps is ~1.1 h, 500k steps ~2.8 h. Use AMP only for the net; keep FK in fp32. ProtoRes itself trained 106 GPU-hours on an M40 for 40k epochs at 41M params [read], so our budget is much smaller; expect weaker result than the paper. Plan a 1-hour smoke run first. Kaggle (P100/T4, 30 h/week) or Colab for sweeps over width (256/384/512) and loss weights.

## 5. Scaffold status (`training/`)
Files: `README.md`, `requirements.txt`, `skeleton.py`, `fk.py`, `model.py`, `benchmark.py`, `tests/test_fk.py`, `tests/test_model.py`.
Environment found: Python 3.13.5, numpy 2.4.6, scipy 1.18.1; torch and onnxruntime were NOT installed. I installed the CPU wheels of torch 2.15.0, onnx 1.23.2, onnxruntime 1.31.1 into a throwaway venv in the agent scratchpad (not the system python, not in repo), about a few hundred MB.
Results: `pytest` 9 passed in 2.65 s (CPU) [ran]; benchmark numbers in section 2.5 [ran].
Not done: data prep scripts, training loop, trained weights, export of quantised model, tests against real SMPL FK.
Note `REST_OFFSETS` are hand-estimated approximations, not verified against SMPL.

## 6. Risks
1. Rest offsets and axis conventions mismatch to AMASS (z-up, SMPL rest-pose orientation) -> silently wrong poses. Mitigation: step 5 regression test.
2. Retargeting to arbitrary rigs: the rig layer, not the net, is the likely source of visible errors (twist, A vs T pose, spine counts, roll). It is the biggest unknown for "any humanoid".
3. Sub-cm accuracy relies on analytic IK post-step, and unreachable targets must be handled (clamp, stretch off).
4. AMASS bias: mostly walking/standing/sports, limited sitting/climbing/stylised poses. Cartoon rigs with different proportions will extrapolate poorly.
5. Licensing: weights from AMASS are non-commercial; SMPL-IK/VPoser code is not copyable; LAFAN1 ND clause. Wrong if later commercial.
6. Python glue inside Blender dominates the 15 ms budget; net is 0.5 ms.
7. Frame-independent model can jitter during a drag; residual-on-current-pose helps, plus the solver can low-pass the output. Untested.
8. 4 GB VRAM is ample for this model; local Windows training might be slowed by FK python loop (could vectorise by depth levels, 7 levels instead of 22 steps).
9. PDF note: Read tool could not render PDFs (no poppler); I extracted text with pypdf (installed into the system python user site, small). Figures/tables were read as text only; the ProtoRes speed/params come from Table 1 text.

## 7. What the user must do or provide
1. Register at AMASS and SMPL (section 3.3), download the listed SMPL-H G subsets and SMPL neutral model to `D:\data\...`. Tell the agent the paths.
2. Decide the repo licence (affects whether Keemap GPL v3 code can be reused and how to word the non-commercial notice for weights).
3. Decide whether weights get published (see section 3.1 caveat) or only the training code.
4. Provide one or two real rigs you animate with (item 4 in PROJECT.md) so the rig layer can be tested.
5. Optional: choose compute: local 3050 for first run, or Kaggle/Colab account.
