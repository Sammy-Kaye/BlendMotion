"""Read the current pose of mapped bones and write FK rotation keyframes back.

Only rotation channels of mapped bones are keyed (plus, optionally, the hips
location). No helper bones, constraints or rig layers are added.
Rotations are *local* pose-bone rotations (what you see in the Rotation
fields), always exchanged as quaternions regardless of the bone's rotation mode.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import bpy
from mathutils import Euler, Quaternion, Vector

from .mapper import RigMapping


@dataclass
class PoseSnapshot:
    rotations: Dict[str, Quaternion] = field(default_factory=dict)   # canonical -> local rotation
    positions: Dict[str, Vector] = field(default_factory=dict)       # canonical -> world head position
    tails: Dict[str, Vector] = field(default_factory=dict)           # canonical -> world tail position
    hips_location: Optional[Vector] = None                           # local location of the hips bone


def get_rotation(pbone) -> Quaternion:
    """Local rotation of a pose bone as a quaternion, whatever its rotation mode."""
    mode = pbone.rotation_mode
    if mode == 'QUATERNION':
        return pbone.rotation_quaternion.copy()
    if mode == 'AXIS_ANGLE':
        a = pbone.rotation_axis_angle
        return Quaternion(Vector((a[1], a[2], a[3])), a[0])
    return pbone.rotation_euler.to_quaternion()


def set_rotation(pbone, q: Quaternion) -> str:
    """Set the local rotation from a quaternion; returns the keyable data path."""
    mode = pbone.rotation_mode
    if mode == 'QUATERNION':
        pbone.rotation_quaternion = q
        return "rotation_quaternion"
    if mode == 'AXIS_ANGLE':
        axis, angle = q.to_axis_angle()
        pbone.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        return "rotation_axis_angle"
    pbone.rotation_euler = q.to_euler(mode, pbone.rotation_euler)
    return "rotation_euler"


def read_pose(obj, mapping: RigMapping) -> PoseSnapshot:
    """Snapshot rotations and world positions of every mapped bone."""
    snap = PoseSnapshot()
    pbones = obj.pose.bones
    mw = obj.matrix_world
    for canon, name in mapping.mapped_items():
        pb = pbones.get(name)
        if pb is None:
            continue
        snap.rotations[canon] = get_rotation(pb)
        snap.positions[canon] = mw @ pb.head
        snap.tails[canon] = mw @ pb.tail
    hips = mapping.bone("hips")
    if hips and hips in pbones:
        snap.hips_location = pbones[hips].location.copy()
    return snap


@contextmanager
def undo_step(label: str):
    """Make everything done in the block one undo step.

    Inside an operator registered with the UNDO option Blender already does
    this; this helper is for code called outside operators (scripts, tests).
    """
    yield
    try:
        bpy.ops.ed.undo_push(message=label)
    except RuntimeError:
        pass


def write_pose(obj, mapping: RigMapping, rotations: Dict[str, Quaternion],
               frame: Optional[float] = None, key: bool = True,
               hips_location: Optional[Vector] = None,
               undo_label: Optional[str] = "BlendMotion: key pose") -> int:
    """Apply local rotations to mapped bones and key them. Returns bones written.

    `undo_label=None` skips pushing an undo step (use inside an operator).
    """
    pbones = obj.pose.bones
    if frame is None:
        frame = bpy.context.scene.frame_current
    count = 0

    def run():
        nonlocal count
        for canon, q in rotations.items():
            name = mapping.bone(canon)
            pb = pbones.get(name) if name else None
            if pb is None:
                continue
            path = set_rotation(pb, q)
            if key:
                pb.keyframe_insert(path, frame=frame, group=pb.name)
            count += 1
        if hips_location is not None:
            name = mapping.bone("hips")
            pb = pbones.get(name) if name else None
            if pb is not None:
                pb.location = hips_location
                if key:
                    pb.keyframe_insert("location", frame=frame, group=pb.name)

    if undo_label is None:
        run()
    else:
        with undo_step(undo_label):
            run()
    return count
