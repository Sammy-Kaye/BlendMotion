"""Regression test on a real production rig (not in the repo, non-commercial licence).

Opens the .blend read-only in the running background Blender (never saves it).
Set HANDLE_REAL_RIG to override the path; the test is skipped if it is missing.
"""
import os
import time

import bpy

from handle.rig import mapper
from handle.rig import canonical as C

DEFAULT = r"D:\Blender\Rigs\GUTENBURG\Rogue_Rig_v3\MR_Rogue_Rig_v3.blend"


def _open():
    path = os.environ.get("HANDLE_REAL_RIG", DEFAULT)
    if not os.path.exists(path):
        print(f"SKIP real rig not found: {path}")
        return False
    t0 = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    print(f"opened real rig in {time.perf_counter() - t0:.1f} s")
    return True


def test_real_rig_mapping():
    if not _open():
        return
    arms = {o.name: o for o in bpy.data.objects if o.type == 'ARMATURE'}
    print("armatures:", {n: len(o.data.bones) for n, o in arms.items()})
    results = {}
    for name in ("META_Rogue", "RIG_Rogue"):
        obj = arms[name]
        before = [b.name for b in obj.data.bones]
        t0 = time.perf_counter()
        m = mapper.map_armature(obj)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n=== {name}: {len(before)} bones, mapped in {dt:.0f} ms ===")
        print(mapper.format_report(m))
        assert [b.name for b in obj.data.bones] == before, "bones must not be renamed"
        results[name] = m
        # hard guarantees
        for canon, bone in m.mapped_items():
            assert not mapper.is_excluded_name(bone), (name, canon, bone)
            low = bone.lower()
            assert "_ik" not in low and not low.startswith("ik_") and "pole" not in low, (name, canon, bone)
        assert not m.required_unmapped, (name, m.required_unmapped)
        assert m.confidence > 0.8, (name, m.confidence)

    meta, rig = results["META_Rogue"], results["RIG_Rogue"]
    # META: deform-style names
    assert meta.bone("hand_l") == "hand.L" and meta.bone("upper_arm_r") == "upperarm.R"
    assert meta.bone("hips") == "pelvis"
    # control rig: FK controls, not DEF/ORG/IK
    for canon in ("upper_arm_l", "lower_arm_l", "hand_l", "upper_leg_r", "lower_leg_r", "foot_r"):
        assert "_fk" in rig.bone(canon), (canon, rig.bone(canon))
    assert not any(b.startswith(("DEF-", "ORG-", "MCH-")) for _, b in rig.mapped_items() if _ in C.CORE_NAMES)
