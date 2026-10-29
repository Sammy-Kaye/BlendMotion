import time

import bpy
from mathutils import Quaternion, Vector

import rigs
from handle.rig import mapper, pose_io


def _setup():
    rigs.reset()
    obj = rigs.build_ue5()
    m = mapper.map_armature(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='POSE')
    return obj, m


def _fcurves(obj):
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    out = []
    for layer in ad.action.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                out.extend(cb.fcurves)
    return out


def test_read_pose_rest():
    obj, m = _setup()
    snap = pose_io.read_pose(obj, m)
    assert set(snap.rotations) == set(m.bones)
    for q in snap.rotations.values():
        assert q.angle < 1e-5
    assert snap.positions["hand_l"].x > 0.5 and snap.positions["hand_r"].x < -0.5
    assert snap.positions["head"].z > snap.positions["hips"].z
    assert abs(snap.positions["hips"].z - 1.0) < 1e-4


def test_write_pose_keys_rotation_only():
    obj, m = _setup()
    obj.pose.bones[m.bone("lower_arm_l")].rotation_mode = 'XYZ'    # mixed modes must work
    rot = {"upper_arm_l": Quaternion((0, 0, 1), 0.5), "lower_arm_l": Quaternion((0, 1, 0), -0.7),
           "head": Quaternion((1, 0, 0), 0.2)}
    n = pose_io.write_pose(obj, m, rot, frame=5)
    assert n == 3
    fcs = _fcurves(obj)
    assert fcs, "no fcurves created"
    assert all("rotation" in fc.data_path for fc in fcs), {fc.data_path for fc in fcs}
    ua = obj.pose.bones[m.bone("upper_arm_l")]
    assert abs(ua.rotation_quaternion.angle - 0.5) < 1e-5
    la = obj.pose.bones[m.bone("lower_arm_l")]
    assert abs(la.rotation_euler.to_quaternion().angle - 0.7) < 1e-5
    q_keys = [fc for fc in fcs if fc.data_path.endswith("rotation_quaternion")]
    assert len(q_keys) == 8   # 2 quaternion bones x 4 channels
    assert all(len(fc.keyframe_points) == 1 and fc.keyframe_points[0].co[0] == 5 for fc in fcs)
    assert len(obj.pose.bones) == len(obj.data.bones)   # no rig layer added
    assert all(len(pb.constraints) == 0 for pb in obj.pose.bones)
    snap = pose_io.read_pose(obj, m)
    assert abs(snap.rotations["head"].angle - 0.2) < 1e-5


def test_undo_reverts_whole_key_pass_in_one_step():
    obj, m = _setup()
    name = obj.name   # object pointers go stale after undo, so look it up by name
    bpy.ops.ed.undo_push(message="before")
    rot = {"upper_arm_l": Quaternion((0, 0, 1), 0.5), "head": Quaternion((1, 0, 0), 0.2)}
    pose_io.write_pose(obj, m, rot, frame=1)
    assert len(_fcurves(bpy.data.objects[name])) == 8
    bpy.ops.ed.undo()   # a single undo must remove every key written by the call
    o2 = bpy.data.objects[name]
    assert len(_fcurves(o2)) == 0, "one undo should remove all keyframes"
    assert o2.pose.bones[m.bone("upper_arm_l")].rotation_quaternion.angle < 1e-6


def test_hips_location_optional():
    obj, m = _setup()
    pose_io.write_pose(obj, m, {}, frame=1, hips_location=Vector((0, 0.1, 0)))
    assert any(fc.data_path.endswith("location") for fc in _fcurves(obj))


def test_read_speed():
    obj, m = _setup()
    t0 = time.perf_counter()
    for _ in range(50):
        pose_io.read_pose(obj, m)
    dt = (time.perf_counter() - t0) / 50 * 1000
    print(f"read_pose ({len(m.bones)} bones): {dt:.2f} ms")
    assert dt < 15
