"""Canonical 22-joint skeleton (SMPL body joints 0-21, no hands/face).

Conventions (must match the handle/rig normalisation layer):
  * Y up, character faces +Z, character's LEFT is +X (SMPL convention).
  * Rest pose is a T-pose; every joint's rest rotation is identity, so a local
    rotation is the rotation away from the T-pose, in the parent frame.
  * Units are metres for a ~1.70 m figure.  Rigs of other sizes are scaled
    uniformly by the rig layer before inference (see docs/ml-research.md).

REST_OFFSETS are an APPROXIMATION of SMPL neutral-body joint offsets, good
enough for scaffolding, tests and benchmarks.  Before real training replace
them with offsets derived from the SMPL neutral model (J_regressor @ v_template)
so AMASS rotations stay consistent.  They are NOT verified against SMPL.
"""
from __future__ import annotations

import numpy as np

JOINT_NAMES = [
    "pelvis",      # 0
    "l_hip",       # 1
    "r_hip",       # 2
    "spine1",      # 3
    "l_knee",      # 4
    "r_knee",      # 5
    "spine2",      # 6
    "l_ankle",     # 7
    "r_ankle",     # 8
    "spine3",      # 9
    "l_foot",      # 10
    "r_foot",      # 11
    "neck",        # 12
    "l_collar",    # 13
    "r_collar",    # 14
    "head",        # 15
    "l_shoulder",  # 16
    "r_shoulder",  # 17
    "l_elbow",     # 18
    "r_elbow",     # 19
    "l_wrist",     # 20
    "r_wrist",     # 21
]
NUM_JOINTS = len(JOINT_NAMES)  # 22

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Offset of each joint from its parent in the T-pose (metres, Y up, left = +X).
REST_OFFSETS = np.array(
    [
        [0.000, 0.000, 0.000],     # pelvis (root; ~0.93 m above floor)
        [0.058, -0.082, -0.002],   # l_hip
        [-0.058, -0.082, -0.002],  # r_hip
        [0.000, 0.108, 0.003],     # spine1
        [0.000, -0.390, 0.000],    # l_knee
        [0.000, -0.390, 0.000],    # r_knee
        [0.000, 0.135, 0.000],     # spine2
        [0.000, -0.400, -0.010],   # l_ankle
        [0.000, -0.400, -0.010],   # r_ankle
        [0.000, 0.052, 0.003],     # spine3
        [0.000, -0.050, 0.120],    # l_foot (toe base)
        [0.000, -0.050, 0.120],    # r_foot
        [0.000, 0.213, -0.040],    # neck
        [0.045, 0.080, -0.010],    # l_collar
        [-0.045, 0.080, -0.010],   # r_collar
        [0.000, 0.065, 0.040],     # head
        [0.090, 0.010, -0.010],    # l_shoulder
        [-0.090, 0.010, -0.010],   # r_shoulder
        [0.260, 0.000, 0.000],     # l_elbow
        [-0.260, 0.000, 0.000],    # r_elbow
        [0.260, 0.000, 0.000],     # l_wrist
        [-0.260, 0.000, 0.000],    # r_wrist
    ],
    dtype=np.float32,
)

# Effector slots exposed to the user (order is part of the model contract).
EFFECTOR_NAMES = ["hips", "head", "l_hand", "r_hand", "l_foot", "r_foot"]
EFFECTOR_JOINTS = [
    JOINT_NAMES.index(n) for n in ("pelvis", "head", "l_wrist", "r_wrist", "l_ankle", "r_ankle")
]
NUM_EFFECTORS = len(EFFECTOR_NAMES)


def _mirror(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


# Left/right joint swap for mirror augmentation (x -> -x).
MIRROR_PERM = [JOINT_NAMES.index(_mirror(n)) for n in JOINT_NAMES]

# Seed aliases for the rig auto-mapper (lower-case, separators stripped).
# The rig layer owns the real mapping; this is only a starting list.
ALIASES = {
    "pelvis": ["hips", "pelvis", "root", "mixamorighips"],
    "spine1": ["spine", "spine1", "spine01", "abdomen"],
    "spine2": ["spine1", "spine2", "spine02", "chest"],
    "spine3": ["spine2", "spine3", "spine03", "upperchest"],
    "neck": ["neck", "neck1"],
    "head": ["head"],
}


def _check() -> None:
    assert len(PARENTS) == NUM_JOINTS == len(REST_OFFSETS)
    for i, p in enumerate(PARENTS):
        assert p < i, "parents must precede children (topological order)"
    assert sorted(MIRROR_PERM) == list(range(NUM_JOINTS))


_check()
