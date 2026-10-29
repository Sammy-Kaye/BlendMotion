"""ProtoRes-lite: stub pose model for drag-to-pose.  UNTRAINED scaffold.

Contract (canonical 22-joint space, metres, Y-up, rig already scaled):
  effectors (B, 6, 11)  per slot: [mask, pos(3), rot6d(6), tolerance]
                        pos is relative to the anchor (centroid of active
                        positional effectors, or current pelvis if none).
  cur_pose  (B, 22*6+3) current local rot6d of every joint (joint 0 = global
                        pelvis rotation) + pelvis position relative to anchor.
  returns   (B, 22*6+3) same layout: new pose = cur_pose + predicted delta.
Residual output means "no effectors" == identity and drags stay temporally
coherent.  The caller orthonormalises rot6d (fk.rot6d_to_matrix).
"""
from __future__ import annotations

import torch
from torch import nn

from skeleton import NUM_EFFECTORS, NUM_JOINTS

EFF_DIM = 11
POSE_DIM = NUM_JOINTS * 6 + 3


class ResBlock(nn.Module):
    """N-BEATS style block used by ProtoRes: returns (backcast residual, forecast)."""

    def __init__(self, width: int, d_out: int, layers: int = 2):
        super().__init__()
        mods = []
        for _ in range(layers):
            mods += [nn.Linear(width, width), nn.ReLU()]
        self.mlp = nn.Sequential(*mods)
        self.back = nn.Linear(width, width)
        self.fore = nn.Linear(width, d_out)

    def forward(self, x):
        h = self.mlp(x)
        return torch.relu(x - self.back(h)), self.fore(h)


class ProtoResLite(nn.Module):
    def __init__(self, tok_width=256, emb=256, dec_width=512, enc_blocks=2, dec_blocks=3, slot_emb=16):
        super().__init__()
        assert tok_width == emb, "proto subtraction needs tok_width == emb"
        self.slot_emb = nn.Parameter(torch.randn(NUM_EFFECTORS, slot_emb) * 0.1)
        self.tok_in = nn.Linear(EFF_DIM + slot_emb, tok_width)
        self.enc = nn.ModuleList([ResBlock(tok_width, emb) for _ in range(enc_blocks)])
        self.pose_enc = nn.Sequential(nn.Linear(POSE_DIM, emb), nn.ReLU(), nn.Linear(emb, emb))
        self.dec_in = nn.Linear(2 * emb, dec_width)
        self.dec = nn.ModuleList([ResBlock(dec_width, dec_width) for _ in range(dec_blocks)])
        self.out = nn.Linear(dec_width, POSE_DIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, effectors, cur_pose):
        B = effectors.shape[0]
        mask = effectors[..., 0:1]                       # (B,S,1)
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)  # (B,1,1)
        x = torch.relu(self.tok_in(torch.cat([effectors, self.slot_emb.expand(B, -1, -1)], -1)))
        e = 0.0
        for i, blk in enumerate(self.enc):
            x, f = blk(x)
            e = e + f
            if i < len(self.enc) - 1:  # proto-residual: subtract pooled pose embedding
                proto = (e * mask).sum(1, keepdim=True) / denom
                x = torch.relu(x - proto / (i + 1.0))
        emb = ((e * mask).sum(1, keepdim=True) / denom).squeeze(1)  # masked mean over slots
        h = torch.relu(self.dec_in(torch.cat([emb, self.pose_enc(cur_pose)], -1)))
        f = 0.0
        for blk in self.dec:
            h, fi = blk(h)
            f = f + fi
        return cur_pose + self.out(torch.relu(f))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
