# Rig mapping (Phase 0)

`handle/rig` maps any humanoid armature onto a fixed canonical skeleton without renaming or touching the rig.

## Canonical skeleton (`handle/rig/canonical.py`)

52 bones: `hips`, `spine`, `chest`, `upper_chest`, `neck`, `head`, and per side (`_l`, `_r`): `shoulder`, `upper_arm`, `lower_arm`, `hand`, `upper_leg`, `lower_leg`, `foot`, `toes`, and 15 finger joints `<thumb|index|middle|ring|pinky>_<1..3>_<side>`.

Required (a rig counts as usable when all are mapped): hips, spine, head, arms (upper/lower/hand), legs (upper/lower/foot). Optional: chest, upper_chest, neck, shoulder, toes, fingers.

Spine convention: rigs with 1, 2 or 3+ spine bones map to spine / spine+chest / spine+chest+upper_chest. With 3+ the slots are first, middle, last (a 5-bone spine `spine_01..05` gives 01, 03, 05). Bones in between are not driven by the solver.

Left is +X, character faces -Y, +Z up (Blender default).

## Mapper (`handle/rig/mapper.py`)

1. Name pass. Names are tokenised (separators, camelCase), rig prefixes (`mixamorig:`, `Bip01`, `CC_Base`, `c_`, `DEF-`, `ORG-`) and the side (`L/R`, `.L`, `_l`, `Left`) are peeled off, and the remaining base word is looked up (`UpLeg`, `calf`, `lowerarm`, `ball`, `ToeBase`, ...). Finger joints need a digit 1..3.
2. Never mapped: IK targets and controllers, pole bones, and MCH / tweak / twist / roll / fix / corrective / master / helper / end bones (see `_HARD`).
3. Preference among duplicates: FK control (tiny tie-break bonus) = plain bone > `DEF-` (x0.85) > `ORG-` (x0.4). Bones driven by a constraint (copy transforms/rotation, IK, tracking...) or a rotation driver are x0.6, because keying them does nothing. Duplicate suffix `.001` is x0.6. A name whose side disagrees with the x position is x0.5. A candidate that is an ancestor of the chosen child bone (or a descendant of the chosen parent) gets a selection bonus and +0.05 confidence.
4. Geometry/topology fallback runs when anything is still unmapped, and only fills empty slots (it is dropped if it disagrees with the name pass about the hips). Hips = bone with an upward branch and two downward branches on opposite x sides; legs, spine, neck/head, arms follow the dominant chains; left/right from x sign; fingers ordered across the palm (thumb = the end finger with the larger gap). Confidence 0.6 (core), 0.5 (neck), 0.45 (fingers).
5. Manual overrides (`overrides={canonical: bone}`, `RigMapping.set_override`) always win and have confidence 1.0, method `manual`.

Confidence per bone is in [0, 1]. `RigMapping.confidence` is the mean over the required bones (unmapped counts 0). Alias-level name matches (`Arm`, `Leg`, `Hip`, `Collar`) score 0.85-0.9; exact words 1.0.

## Files

- `save_mapping(mapping, path)` / `load_mapping(path)`: JSON `{version, armature, bones: {canonical: {bone, confidence, method, note?}}, unmapped}`. Unknown canonical keys are ignored on load; `missing_in(obj)` lists bones that no longer exist.
- The add-on stores the mapping as the object custom property `handle_mapping` (JSON) after "Detect Rig".

## Pose IO (`handle/rig/pose_io.py`)

- `read_pose(obj, mapping)` returns local rotations (quaternions, whatever the bone's rotation mode) and world head/tail positions per canonical bone.
- `write_pose(obj, mapping, rotations, frame=None, hips_location=None)` sets rotations and keys only the rotation channel of the bone's own mode (plus hips location if given). It adds no bones, constraints or layers and is one undo step (`undo_label=None` inside an operator, which owns the step via `bl_options={'UNDO'}`).

## Tests

`blender -b --factory-startup --python-exit-code 1 --python tests/run.py` (add `-- --no-real` to skip the production-rig test, `-- -k substring` to filter). The real-rig test opens `HANDLE_REAL_RIG` (default `D:\Blender\Rigs\GUTENBURG\Rogue_Rig_v3\MR_Rogue_Rig_v3.blend`) read-only and is skipped when absent. That rig is non-commercially licensed: keep `*.blend` out of git.
