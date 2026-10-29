import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import EFF_DIM, POSE_DIM, ProtoResLite, count_params  # noqa: E402
from skeleton import NUM_EFFECTORS  # noqa: E402


def test_shapes_and_identity_init():
    m = ProtoResLite().eval()
    eff = torch.randn(3, NUM_EFFECTORS, EFF_DIM)
    eff[..., 0] = 1.0
    pose = torch.randn(3, POSE_DIM)
    out = m(eff, pose)
    assert out.shape == pose.shape
    assert torch.allclose(out, pose)  # zero-init head => identity before training


def test_mask_ignores_inactive_slots():
    torch.manual_seed(0)
    m = ProtoResLite().eval()
    for p in m.out.parameters():
        torch.nn.init.normal_(p, std=0.05)
    eff = torch.randn(1, NUM_EFFECTORS, EFF_DIM)
    eff[..., 0] = 0.0
    eff[:, 2, 0] = 1.0
    eff2 = eff.clone()
    eff2[:, 3:, 1:] = torch.randn_like(eff2[:, 3:, 1:])  # change only masked slots
    pose = torch.randn(1, POSE_DIM)
    assert torch.allclose(m(eff, pose), m(eff2, pose), atol=1e-5)


def test_param_budget():
    assert count_params(ProtoResLite()) < 5_000_000  # < 20 MB fp32 with huge margin
