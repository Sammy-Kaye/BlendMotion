import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fk import SkeletonFK, axis_angle_to_matrix, matrix_to_rot6d, rot6d_to_matrix  # noqa: E402
from skeleton import JOINT_NAMES, MIRROR_PERM, NUM_JOINTS, PARENTS, REST_OFFSETS  # noqa: E402


def test_rot6d_roundtrip():
    torch.manual_seed(0)
    m = axis_angle_to_matrix(torch.randn(64, 3))
    m2 = rot6d_to_matrix(matrix_to_rot6d(m))
    assert torch.allclose(m, m2, atol=1e-5)
    assert torch.allclose(m2 @ m2.transpose(-1, -2), torch.eye(3).expand_as(m2), atol=1e-5)
    assert torch.allclose(torch.linalg.det(m2), torch.ones(64), atol=1e-5)


def test_identity_pose_is_rest_pose():
    fk = SkeletonFK()
    eye = torch.eye(3).repeat(2, NUM_JOINTS, 1, 1)
    pos, _ = fk(eye, torch.zeros(2, 3))
    ref = torch.zeros(NUM_JOINTS, 3)
    for j in range(1, NUM_JOINTS):
        ref[j] = ref[PARENTS[j]] + torch.as_tensor(REST_OFFSETS[j])
    assert torch.allclose(pos[0], ref, atol=1e-6)
    assert ref[:, 1].min() < -0.8  # ankles/feet ~0.9 m below pelvis


def test_bone_lengths_preserved_under_random_rotation():
    torch.manual_seed(1)
    fk = SkeletonFK()
    rot = axis_angle_to_matrix(torch.randn(8, NUM_JOINTS, 3))
    pos, _ = fk(rot, torch.randn(8, 3))
    for j in range(1, NUM_JOINTS):
        d = (pos[:, j] - pos[:, PARENTS[j]]).norm(dim=-1)
        want = torch.as_tensor(REST_OFFSETS[j]).norm().expand(8)
        assert torch.allclose(d, want, atol=1e-5)


def test_known_rotation_moves_hand():
    fk = SkeletonFK()
    eye = torch.eye(3).repeat(1, NUM_JOINTS, 1, 1)
    rot = eye.clone()
    s = JOINT_NAMES.index("l_shoulder")
    w = JOINT_NAMES.index("l_wrist")
    # +90 deg about Z swings the +X arm up to +Y
    rot[0, s] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2]))
    pos, _ = fk(rot, torch.zeros(1, 3))
    arm = pos[0, w] - pos[0, s]
    assert arm[1] > 0.5 and abs(arm[0]) < 1e-5


def test_gradients_flow():
    fk = SkeletonFK()
    x = torch.randn(4, NUM_JOINTS, 6, requires_grad=True)
    pos = fk.effector_positions(rot6d_to_matrix(x), torch.zeros(4, 3))
    pos.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mirror_perm_is_involution():
    assert [MIRROR_PERM[MIRROR_PERM[i]] for i in range(NUM_JOINTS)] == list(range(NUM_JOINTS))
    assert JOINT_NAMES[MIRROR_PERM[1]] == "r_hip"
