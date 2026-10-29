"""Synthetic humanoid armatures for the tests, built purely in code.

A neutral template (T-pose, 1.8 m, faces -Y, character's left is +X) is
rendered through different naming schemes: Mixamo, UE5 (with IK/twist bones),
game style with dotted sides, a layered Rigify-like rig (FK/IK/DEF/ORG), and
a nameless one ("Bone.001") to exercise the geometry fallback.
"""
import random

import bpy

SPINE = 3
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_Y = {"thumb": -0.075, "index": -0.025, "middle": -0.005, "ring": 0.015, "pinky": 0.035}


def _is_finger(part):
    return part.split("_")[0] in FINGERS


def template(spine_n=SPINE):
    """Ordered list of (canon_name, parent_canon, head, tail).

    Nub (end) bones use names like 'nub:head', 'nub:toes_l', 'nub:index_3_l'.
    """
    rows = []

    def add(name, parent, head, tail):
        rows.append((name, parent, tuple(head), tuple(tail)))

    add("hips", None, (0, 0, 1.0), (0, 0, 1.1))
    names = ["spine", "chest", "upper_chest"]
    prev = "hips"
    for i in range(spine_n):
        z0 = 1.1 + 0.13 * i
        add(names[i], prev, (0, 0, z0), (0, 0, z0 + 0.13))
        prev = names[i]
    top = 1.1 + 0.13 * spine_n
    add("neck", prev, (0, 0, top), (0, 0, top + 0.1))
    add("head", "neck", (0, 0, top + 0.1), (0, 0, top + 0.25))
    add("nub:head", "head", (0, 0, top + 0.25), (0, 0, top + 0.27))
    for s, sg in (("l", 1), ("r", -1)):
        az = top - 0.05
        add(f"shoulder_{s}", prev, (sg * 0.03, 0, az), (sg * 0.15, 0, az))
        add(f"upper_arm_{s}", f"shoulder_{s}", (sg * 0.15, 0, az), (sg * 0.42, 0, az))
        add(f"lower_arm_{s}", f"upper_arm_{s}", (sg * 0.42, 0, az), (sg * 0.68, 0, az))
        add(f"hand_{s}", f"lower_arm_{s}", (sg * 0.68, 0, az), (sg * 0.76, 0, az))
        for f, y in FINGER_Y.items():
            z = az - (0.02 if f == "thumb" else 0.0)
            x0 = 0.72 if f == "thumb" else 0.77
            prev_f = f"hand_{s}"
            for k in (1, 2, 3):
                x1 = x0 + 0.035
                add(f"{f}_{k}_{s}", prev_f, (sg * x0, y, z), (sg * x1, y, z))
                prev_f = f"{f}_{k}_{s}"
                x0 = x1
            add(f"nub:{f}_3_{s}", prev_f, (sg * x0, y, z), (sg * (x0 + 0.01), y, z))
        add(f"upper_leg_{s}", "hips", (sg * 0.09, 0, 1.0), (sg * 0.09, 0, 0.55))
        add(f"lower_leg_{s}", f"upper_leg_{s}", (sg * 0.09, 0, 0.55), (sg * 0.09, 0, 0.1))
        add(f"foot_{s}", f"lower_leg_{s}", (sg * 0.09, 0, 0.1), (sg * 0.09, -0.12, 0.03))
        add(f"toes_{s}", f"foot_{s}", (sg * 0.09, -0.12, 0.03), (sg * 0.09, -0.2, 0.02))
        add(f"nub:toes_{s}", f"toes_{s}", (sg * 0.09, -0.2, 0.02), (sg * 0.09, -0.21, 0.02))
    return rows


def build_armature(name, rows, shuffle_seed=None):
    """rows: (bone_name, parent_name or None, head, tail). Returns the object."""
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    order = list(rows)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    eb = arm.edit_bones
    for bname, _, head, tail in order:
        b = eb.new(bname)
        b.head, b.tail = head, tail
        b.use_deform = True
    for bname, parent, _, _ in order:
        if parent:
            eb[bname].parent = eb[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def _render(rows, namer, extra=()):
    out = []
    for canon, parent, head, tail in rows:
        out.append((namer(canon), namer(parent) if parent else None, head, tail))
    out.extend(extra)
    return out


def expected_names(namer, spine_n=SPINE):
    """canonical -> bone name for every canonical bone present in the template."""
    from handle.rig import canonical as C
    present = {r[0] for r in template(spine_n)}
    return {c: namer(c) for c in C.NAMES if c in present}


# ---- namers ---------------------------------------------------------------

_MX = {"hips": "Hips", "spine": "Spine", "chest": "Spine1", "upper_chest": "Spine2",
       "neck": "Neck", "head": "Head", "shoulder": "Shoulder", "upper_arm": "Arm",
       "lower_arm": "ForeArm", "hand": "Hand", "upper_leg": "UpLeg", "lower_leg": "Leg",
       "foot": "Foot", "toes": "ToeBase"}


def mixamo(canon):
    p = "mixamorig:"
    if canon.startswith("nub:"):
        c = canon[4:]
        if c == "head":
            return p + "HeadTop_End"
        side = "Left" if c.endswith("_l") else "Right"
        if c.startswith("toes"):
            return p + side + "Toe_End"
        return f"{p}{side}Hand{c.split('_')[0].capitalize()}4"
    if canon[-2:] in ("_l", "_r"):
        side = "Left" if canon.endswith("_l") else "Right"
        part = canon[:-2]
        if _is_finger(part):
            f, k = part.split("_")
            return f"{p}{side}Hand{f.capitalize()}{k}"
        return f"{p}{side}{_MX[part]}"
    return p + _MX[canon]


_UE = {"hips": "pelvis", "spine": "spine_01", "chest": "spine_02", "upper_chest": "spine_03",
       "neck": "neck_01", "head": "head", "shoulder": "clavicle", "upper_arm": "upperarm",
       "lower_arm": "lowerarm", "hand": "hand", "upper_leg": "thigh", "lower_leg": "calf",
       "foot": "foot", "toes": "ball"}


def ue5(canon):
    if canon.startswith("nub:"):
        return "nub_" + canon[4:]
    if canon[-2:] in ("_l", "_r"):
        part, s = canon[:-2], canon[-1]
        if _is_finger(part):
            f, k = part.split("_")
            return f"{f}_0{k}_{s}"
        return f"{_UE[part]}_{s}"
    return _UE[canon]


_GD = {"hips": "pelvis", "spine": "spine_01", "chest": "spine_02", "upper_chest": "spine_03",
       "neck": "neck_01", "head": "head", "shoulder": "shoulder", "upper_arm": "upperarm",
       "lower_arm": "forearm", "hand": "hand", "upper_leg": "thigh", "lower_leg": "shin",
       "foot": "foot", "toes": "toe"}


def game_dotted(canon):
    """pelvis / spine_01 / upperarm.L style."""
    if canon.startswith("nub:"):
        return "end_" + canon[4:]
    if canon[-2:] in ("_l", "_r"):
        part, s = canon[:-2], canon[-1].upper()
        if _is_finger(part):
            f, k = part.split("_")
            return f"{f}_0{k}.{s}"
        return f"{_GD[part]}.{s}"
    return _GD[canon]


def layered_name(kind, canon):
    base = game_dotted(canon)
    stem, s = base, ""
    if base[-2:] in (".L", ".R"):
        stem, s = base[:-2], base[-2:]
    return {"fk": f"{stem}_fk{s}", "def": f"DEF-{stem}{s}", "org": f"ORG-{stem}{s}",
            "ik": f"{stem}_ik{s}"}[kind]


# ---- builders ---------------------------------------------------------------

def build_mixamo():
    return build_armature("Mixamo_Test", _render(template(), mixamo))


def ue_extras():
    """IK, pole, twist and root bones as on the UE5 mannequin."""
    ex = [("root", None, (0, 0, 0), (0, 0, 0.1)),
          ("ik_foot_root", "root", (0, 0, 0), (0, 0, 0.1)),
          ("ik_hand_root", "root", (0, 0, 0.2), (0, 0, 0.3))]
    for s, sg in (("l", 1), ("r", -1)):
        ex += [(f"ik_foot_{s}", "ik_foot_root", (sg * 0.09, 0, 0.1), (sg * 0.09, -0.1, 0.03)),
               (f"ik_hand_{s}", "ik_hand_root", (sg * 0.68, 0, 1.7), (sg * 0.76, 0, 1.7)),
               (f"pole_knee_{s}", "root", (sg * 0.09, -0.5, 0.55), (sg * 0.09, -0.6, 0.55)),
               (f"lowerarm_twist_01_{s}", f"lowerarm_{s}", (sg * 0.5, 0, 1.75), (sg * 0.55, 0, 1.75)),
               (f"upperarm_twist_01_{s}", f"upperarm_{s}", (sg * 0.3, 0, 1.75), (sg * 0.35, 0, 1.75)),
               (f"calf_twist_01_{s}", f"calf_{s}", (sg * 0.09, 0, 0.3), (sg * 0.09, 0, 0.25))]
    return ex


def build_ue5():
    rows = _render(template(), ue5, ue_extras())
    rows = [(n, ("root" if n == "pelvis" else p), h, t) for n, p, h, t in rows]
    return build_armature("UE5_Test", rows)


def build_game_dotted():
    ex = [("root", None, (0, 0, 0), (0, 0, 0.1)),
          ("ik_foot_root", "root", (0, 0, 0), (0, 0, 0.1)),
          ("ik_foot.L", "ik_foot_root", (0.09, 0, 0.1), (0.09, -0.1, 0.03)),
          ("ik_foot.R", "ik_foot_root", (-0.09, 0, 0.1), (-0.09, -0.1, 0.03)),
          ("ik_hand.L", "root", (0.68, 0, 1.7), (0.76, 0, 1.7)),
          ("ik_hand.R", "root", (-0.68, 0, 1.7), (-0.76, 0, 1.7)),
          ("forearm_twist_01.L", "forearm.L", (0.5, 0, 1.75), (0.55, 0, 1.75)),
          ("Shoulder_Fix_1.L", "shoulder.L", (0.1, 0, 1.75), (0.12, 0, 1.75))]
    rows = _render(template(), game_dotted, ex)
    rows = [(n, ("root" if n == "pelvis" else p), h, t) for n, p, h, t in rows]
    return build_armature("Dotted_Test", rows)


LAYER_PARTS = ("shoulder", "upper_arm", "lower_arm", "hand", "upper_leg", "lower_leg", "foot", "toes")


def build_layered():
    """Rigify-like: ORG + DEF (copy-transforms driven) + FK controls + IK copies."""
    main = []
    for canon, parent, head, tail in template():
        base = canon[:-2] if canon[-2:] in ("_l", "_r") else canon
        if canon.startswith("nub:") or _is_finger(base):
            continue
        main.append((canon, parent, head, tail))
    rows = [("root", None, (0, 0, 0), (0, 0, 0.1))]
    def_to_fk = {}
    for kind in ("org", "fk", "def", "ik"):
        for canon, parent, head, tail in main:
            base = canon[:-2] if canon[-2:] in ("_l", "_r") else canon
            if kind == "ik":
                if base not in ("upper_arm", "lower_arm", "hand", "upper_leg", "lower_leg", "foot"):
                    continue
                pbase = parent[:-2] if parent and parent[-2:] in ("_l", "_r") else parent
                par = layered_name("ik", parent) if pbase in ("upper_arm", "lower_arm", "upper_leg", "lower_leg") else "root"
            else:
                par = layered_name(kind, parent) if parent else "root"
            name = layered_name(kind, canon)
            rows.append((name, par, head, tail))
            if kind == "def":
                def_to_fk[name] = layered_name("fk", canon)
    obj = build_armature("Layered_Test", rows)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='POSE')
    for pb in obj.pose.bones:
        if pb.name in def_to_fk:
            c = pb.constraints.new('COPY_TRANSFORMS')
            c.target = obj
            c.subtarget = def_to_fk[pb.name]
        elif pb.name.startswith("ORG-"):
            c = pb.constraints.new('COPY_ROTATION')
            c.target = obj
            c.subtarget = "root"
    bpy.ops.object.mode_set(mode='OBJECT')
    for b in obj.data.bones:
        b.use_deform = b.name.startswith("DEF-")
    return obj


def build_nameless(seed=3):
    """Same body, meaningless names, shuffled creation order.

    Returns (obj, truth) where truth maps canonical -> bone name."""
    rows = template()
    shuffled = random.Random(seed).sample(rows, len(rows))
    idx = {r[0]: f"Bone.{i:03d}" for i, r in enumerate(shuffled)}
    rn = _render(rows, lambda c: idx[c])
    obj = build_armature("Nameless_Test", rn, shuffle_seed=seed)
    truth = {c: n for c, n in idx.items() if not c.startswith("nub:")}
    return obj, truth


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
