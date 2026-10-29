"""Differentiable forward kinematics in PyTorch for the canonical skeleton."""
from __future__ import annotations

import torch

from skeleton import EFFECTOR_JOINTS, PARENTS, REST_OFFSETS


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt (Zhou et al. 2019).

    The 6D vector is the first two COLUMNS of the rotation matrix.
    """
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): first two columns."""
    return torch.cat((m[..., :, 0], m[..., :, 1]), dim=-1)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) Rodrigues.  Used to ingest AMASS poses."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack((zero, -kz, ky, kz, zero, -kx, -ky, kx, zero), dim=-1)
    K = K.reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


class SkeletonFK(torch.nn.Module):
    """Batched FK: local rot (B,J,3,3) + root pos (B,3) -> global pos/rot.

    `offsets` can be overridden per call, shape (B,J,3), to train with varied
    bone lengths (proportion augmentation).
    """

    def __init__(self, parents=PARENTS, offsets=REST_OFFSETS):
        super().__init__()
        self.parents = list(parents)
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.float32))
        self.register_buffer("eff_idx", torch.tensor(EFFECTOR_JOINTS, dtype=torch.long))

    def forward(self, local_rot, root_pos, offsets=None):
        if offsets is None:
            offsets = self.offsets.expand(local_rot.shape[0], -1, -1)
        J = local_rot.shape[1]
        gr = [local_rot[:, 0]]
        gp = [root_pos]
        for j in range(1, J):
            p = self.parents[j]
            gp.append(gp[p] + (gr[p] @ offsets[:, j].unsqueeze(-1)).squeeze(-1))
            gr.append(gr[p] @ local_rot[:, j])
        return torch.stack(gp, 1), torch.stack(gr, 1)

    def effector_positions(self, local_rot, root_pos, offsets=None):
        pos, _ = self.forward(local_rot, root_pos, offsets)
        return pos[:, self.eff_idx]
