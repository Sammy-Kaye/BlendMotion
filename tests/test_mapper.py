import json
import os
import tempfile
import time

import bpy

import rigs
from handle.rig import canonical as C
from handle.rig import mapper


def _check_full(obj, namer, min_conf=0.9, methods=("name",)):
    m = mapper.map_armature(obj)
    exp = rigs.expected_names(namer)
    got = {k: v.bone for k, v in m.bones.items()}
    wrong = {k: (got.get(k), v) for k, v in exp.items() if got.get(k) != v}
    assert not wrong, f"wrong/missing: {wrong}\n{mapper.format_report(m)}"
    assert not m.required_unmapped, m.required_unmapped
    assert m.confidence >= min_conf, m.confidence
    assert set(got) == set(exp), set(got) ^ set(exp)
    for k, e in m.bones.items():
        assert e.method in methods, (k, e)
    return m


def test_mixamo():
    rigs.reset()
    obj = rigs.build_mixamo()
    m = _check_full(obj, rigs.mixamo, 0.97)   # 'Arm'/'Leg' are alias-level matches (<1.0)
    assert m.bone("hand_l") == "mixamorig:LeftHand"
    assert m.bone("upper_leg_r") == "mixamorig:RightUpLeg"
    assert m.bone("lower_leg_l") == "mixamorig:LeftLeg"
    assert m.bone("pinky_3_r") == "mixamorig:RightHandPinky3"
    assert m.unmapped == [], m.unmapped
    assert "mixamorig:Hips" in obj.data.bones   # names untouched


def test_ue5_ignores_ik_and_twist():
    rigs.reset()
    obj = rigs.build_ue5()
    m = _check_full(obj, rigs.ue5, 0.99)
    for canon, name in m.mapped_items():
        assert "ik_" not in name and "twist" not in name and "pole" not in name, (canon, name)
    assert m.bone("hand_l") == "hand_l" and m.bone("lower_arm_r") == "lowerarm_r"
    assert m.bone("toes_l") == "ball_l"


def test_game_dotted_style():
    rigs.reset()
    obj = rigs.build_game_dotted()
    m = _check_full(obj, rigs.game_dotted, 0.99)
    assert m.bone("hand_l") == "hand.L" and m.bone("upper_arm_r") == "upperarm.R"
    assert m.bone("shoulder_l") == "shoulder.L"
    for _, name in m.mapped_items():
        assert not name.startswith("ik_") and "Fix" not in name and "twist" not in name


def test_layered_prefers_fk_never_ik():
    rigs.reset()
    obj = rigs.build_layered()
    m = mapper.map_armature(obj)
    print(mapper.format_report(m))
    for canon in ("upper_arm_l", "lower_arm_l", "hand_l", "upper_leg_r", "lower_leg_r", "foot_l",
                  "toes_l", "hips", "spine", "neck", "head"):
        name = m.bone(canon)
        assert name is not None, canon
        assert "_fk" in name, (canon, name)
    for _, name in m.mapped_items():
        assert "_ik" not in name and not name.startswith(("DEF-", "ORG-")), name
    # DEF-only variant: with the FK controls removed, deform bones are used instead
    rigs.reset()
    obj = rigs.build_layered()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    for eb in list(obj.data.edit_bones):
        if "_fk" in eb.name or "_ik" in eb.name or eb.name.startswith("ORG-"):
            obj.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode='OBJECT')
    m2 = mapper.map_armature(obj)
    assert m2.bone("hand_l") == "DEF-hand.L", m2.bone("hand_l")
    assert m2.bones["hand_l"].confidence < m.bones["hand_l"].confidence


def test_name_parser_corners():
    p = mapper.parse_name
    assert p("mixamorig:LeftHandIndex2").side == "l" and p("mixamorig:LeftHandIndex2").nums == (2,)
    assert p("Bip01 R Thigh").side == "r" and p("Bip01 R Thigh").base == "thigh"
    assert p("CC_Base_L_Upperarm").base == "upperarm"
    assert p("c_thigh_fk.l").side == "l" and "fk" in p("c_thigh_fk.l").flags
    for bad in ("ik_foot.L", "IK_Hand_L", "hand_ik.R", "VIS_thigh_ik_pole.L", "MCH-hand_fk.L",
                "forearm_twist_02.L", "upperarm_tweak.L", "index_01_master.L", "RightToe_End",
                "ikTargetHand", "Shoulder_Fix_3.R", "pole_knee_l", "elbow_target.L"):
        assert mapper.is_excluded_name(bad), bad
    for good in ("upperarm.L", "Hips", "spine_03", "LeftToeBase", "thigh_fk.R", "DEF-hand.L", "head"):
        assert not mapper.is_excluded_name(good), good
    assert not mapper.is_excluded_name("Spike_Spine")   # 'ik' inside a word is not IK


def test_geometry_fallback_on_nameless_rig():
    rigs.reset()
    obj, truth = rigs.build_nameless()
    m = mapper.map_armature(obj)
    print(mapper.format_report(m))
    exp = {c: n for c, n in truth.items() if c in C.BY_NAME}
    wrong = {k: (m.bone(k), v) for k, v in exp.items() if m.bone(k) != v}
    assert not wrong, wrong
    assert all(e.method == "geometry" for e in m.bones.values())
    assert not m.required_unmapped
    assert 0.4 < m.confidence < 0.9, m.confidence   # lower than name matching
    # independent of bone creation order / names
    rigs.reset()
    obj2, truth2 = rigs.build_nameless(seed=11)
    m2 = mapper.map_armature(obj2)
    wrong2 = {k: (m2.bone(k), v) for k, v in truth2.items() if m2.bone(k) != v}
    assert not wrong2, wrong2


def test_geometry_fills_gaps_left_by_names():
    rigs.reset()
    obj = rigs.build_mixamo()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    obj.data.edit_bones["mixamorig:Neck"].name = "mixamorig:Weird01"
    bpy.ops.object.mode_set(mode='OBJECT')
    m = mapper.map_armature(obj)
    assert m.bone("neck") == "mixamorig:Weird01", m.bone("neck")
    assert m.bones["neck"].method == "geometry"
    assert m.bones["hand_l"].method == "name"


def test_json_roundtrip_and_overrides():
    rigs.reset()
    obj = rigs.build_mixamo()
    m = mapper.map_armature(obj, overrides={"toes_l": "mixamorig:LeftFoot"})
    assert m.bones["toes_l"].method == "manual" and m.bone("toes_l") == "mixamorig:LeftFoot"
    path = os.path.join(tempfile.mkdtemp(), "map.json")
    mapper.save_mapping(m, path)
    d = json.load(open(path, encoding="utf-8"))
    assert d["version"] == 1 and d["bones"]["hips"]["bone"] == "mixamorig:Hips"
    m2 = mapper.load_mapping(path)
    assert {k: (v.bone, v.method) for k, v in m2.bones.items()} == \
           {k: (v.bone, v.method) for k, v in m.bones.items()}
    assert abs(m2.confidence - m.confidence) < 1e-3
    assert m2.missing_in(obj) == []
    m2.set_override("hips", "nonexistent")
    assert m2.missing_in(obj) == ["hips"]
    d["bones"]["tail_9"] = {"bone": "x"}   # unknown keys from newer files are tolerated
    assert "tail_9" not in mapper.RigMapping.from_dict(d).bones


def test_mapping_speed():
    rigs.reset()
    obj = rigs.build_ue5()
    infos = mapper.bone_infos(obj)
    t0 = time.perf_counter()
    for _ in range(20):
        mapper.map_bones(infos)
    dt = (time.perf_counter() - t0) / 20 * 1000
    print(f"map_bones on {len(infos)} bones: {dt:.2f} ms")
    assert dt < 50
