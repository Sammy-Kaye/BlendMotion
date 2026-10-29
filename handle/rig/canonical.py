"""Canonical humanoid skeleton definition.

Pure Python (no bpy), so it can be imported anywhere.

Naming: ``<part>_<side>`` with side ``l`` / ``r``; centre bones have no side.
Fingers: ``<finger>_<n>_<side>`` with n in 1..3 (1 = knuckle/proximal).

Spine convention: rigs have anywhere from 1 to 5+ spine bones. They are
mapped onto three slots, ``spine`` (lowest), ``chest`` (middle) and
``upper_chest`` (highest). With 1 bone: spine. With 2: spine, chest.
With 3+: spine = first, chest = middle, upper_chest = last. Bones in between
are simply not driven by the solver.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("l", "r")


@dataclass(frozen=True)
class CanonicalBone:
    name: str
    parent: Optional[str]
    side: Optional[str]      # 'l', 'r' or None
    group: str               # core, spine, head, arm, leg, hand
    required: bool = True    # required for a rig to count as fully mapped
    part: str = ""           # side-less part name, e.g. 'upper_arm'


def _build() -> Tuple[CanonicalBone, ...]:
    out: List[CanonicalBone] = []

    def add(part, parent_part, group, required=True, side=None):
        name = part if side is None else f"{part}_{side}"
        parent = None
        if parent_part is not None:
            parent = parent_part if side is None or parent_part in _CENTRE else f"{parent_part}_{side}"
        out.append(CanonicalBone(name, parent, side, group, required, part))

    _CENTRE = {"hips", "spine", "chest", "upper_chest", "neck", "head"}

    add("hips", None, "core")
    add("spine", "hips", "spine")
    add("chest", "spine", "spine", required=False)
    add("upper_chest", "chest", "spine", required=False)
    add("neck", "upper_chest", "head", required=False)
    add("head", "neck", "head")
    for s in SIDES:
        add("shoulder", "upper_chest", "arm", required=False, side=s)
        add("upper_arm", "shoulder", "arm", side=s)
        add("lower_arm", "upper_arm", "arm", side=s)
        add("hand", "lower_arm", "arm", side=s)
        add("upper_leg", "hips", "leg", side=s)
        add("lower_leg", "upper_leg", "leg", side=s)
        add("foot", "lower_leg", "leg", side=s)
        add("toes", "foot", "leg", required=False, side=s)
        for f in FINGERS:
            prev = "hand"
            for i in (1, 2, 3):
                part = f"{f}_{i}"
                add(part, prev, "hand", required=False, side=s)
                prev = part
    return tuple(out)


BONES: Tuple[CanonicalBone, ...] = _build()
BY_NAME: Dict[str, CanonicalBone] = {b.name: b for b in BONES}
NAMES: Tuple[str, ...] = tuple(b.name for b in BONES)
REQUIRED: Tuple[str, ...] = tuple(b.name for b in BONES if b.required)
# Bones the Phase 1 solver drives directly (end effectors and their chains).
CORE_NAMES: Tuple[str, ...] = tuple(
    b.name for b in BONES if b.group in ("core", "spine", "head", "arm", "leg")
)


def mirror(name: str) -> str:
    """Return the opposite-side canonical name (centre bones map to themselves)."""
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return name


def parent_of(name: str) -> Optional[str]:
    """Canonical parent. Missing optional parents are resolved by `resolved_parent`."""
    return BY_NAME[name].parent


def resolved_parent(name: str, present) -> Optional[str]:
    """Nearest canonical ancestor that is in `present` (skips unmapped optional bones)."""
    p = BY_NAME[name].parent
    while p is not None and p not in present:
        p = BY_NAME[p].parent
    return p
