"""Auto-map an armature's bones onto the canonical skeleton without renaming.

Two stages:

1. Name matching. Bone names are tokenised (separators and camelCase), the
   side (L/R) and rig prefixes (``mixamorig:``, ``DEF-``, ``ORG-``, ``c_``)
   are peeled off, and the remaining base word is looked up in alias tables.
   IK targets, pole bones, mechanism, tweak, twist and helper bones are never
   mapped. Among duplicates the best control wins: FK controls beat plain
   bones, plain bones beat DEF-, those beat ORG-; bones that are driven by
   constraints/drivers are penalised because keyframing them has no effect.
   A small topology bonus prefers a candidate that sits above its mapped
   child (e.g. the clavicle that is really the parent of the upper arm).

2. Geometry/topology fallback for whatever required bones are still unmapped:
   uses the bone hierarchy and rest-pose positions (hips = bone with an up
   branch and two down branches, legs/arms/spine/neck/head by walking the
   dominant chains, left/right by x sign, fingers ordered across the palm).
   Convention: character faces -Y, +Z is up, character's left is +X.

The core (`map_bones`) is pure Python over `BoneInfo`, so it is fast and
testable without Blender; `map_armature` extracts the data from an object.
"""
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import canonical as C

MAPPING_VERSION = 1

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class BoneInfo:
    name: str
    parent: int                      # index into the list, -1 for roots
    head: Tuple[float, float, float]  # rest position, world space
    tail: Tuple[float, float, float]
    deform: bool = False
    driven: bool = False             # rotation overridden by constraint/driver


@dataclass
class BoneMatch:
    bone: str
    confidence: float
    method: str          # 'name', 'geometry', 'manual'
    note: str = ""


@dataclass
class RigMapping:
    armature: str = ""
    bones: Dict[str, BoneMatch] = field(default_factory=dict)

    # -- queries ----------------------------------------------------------
    def bone(self, canon: str) -> Optional[str]:
        m = self.bones.get(canon)
        return m.bone if m else None

    def confidence_of(self, canon: str) -> float:
        m = self.bones.get(canon)
        return m.confidence if m else 0.0

    @property
    def unmapped(self) -> List[str]:
        return [n for n in C.NAMES if n not in self.bones]

    @property
    def required_unmapped(self) -> List[str]:
        return [n for n in C.REQUIRED if n not in self.bones]

    @property
    def confidence(self) -> float:
        """Mean confidence over required bones (unmapped ones count as 0)."""
        req = C.REQUIRED
        return sum(self.confidence_of(n) for n in req) / len(req)

    @property
    def is_usable(self) -> bool:
        return not self.required_unmapped

    def mapped_items(self):
        """(canonical, bone name) for every mapped bone, in canonical order."""
        return [(n, self.bones[n].bone) for n in C.NAMES if n in self.bones]

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": MAPPING_VERSION,
            "armature": self.armature,
            "bones": {
                n: {"bone": m.bone, "confidence": round(m.confidence, 4),
                    "method": m.method, **({"note": m.note} if m.note else {})}
                for n, m in ((n, self.bones[n]) for n in C.NAMES if n in self.bones)
            },
            "unmapped": self.unmapped,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RigMapping":
        if d.get("version", 1) > MAPPING_VERSION:
            raise ValueError("mapping file is newer than this add-on")
        m = cls(armature=d.get("armature", ""))
        for n, e in d.get("bones", {}).items():
            if n not in C.BY_NAME:
                continue  # tolerate files from a newer canonical set
            m.bones[n] = BoneMatch(e["bone"], float(e.get("confidence", 1.0)),
                                   e.get("method", "manual"), e.get("note", ""))
        return m

    def missing_in(self, obj) -> List[str]:
        """Canonical names whose mapped bone does not exist on `obj`."""
        have = obj.data.bones
        return [n for n, m in self.bones.items() if m.bone not in have]

    def set_override(self, canon: str, bone: Optional[str]):
        """Manual override; `bone=None` removes the entry."""
        if canon not in C.BY_NAME:
            raise KeyError(canon)
        if bone is None:
            self.bones.pop(canon, None)
        else:
            self.bones[canon] = BoneMatch(bone, 1.0, "manual")


def save_mapping(mapping: RigMapping, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping.to_dict(), f, indent=2)


def load_mapping(path: str) -> RigMapping:
    with open(path, "r", encoding="utf-8") as f:
        return RigMapping.from_dict(json.load(f))


# --------------------------------------------------------------------------
# Name parsing
# --------------------------------------------------------------------------

_DUP = re.compile(r"\.\d{3}$")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")

# Leading tokens that are rig-system prefixes, not part of the bone's meaning.
_PREFIX = {"mixamorig", "bip", "biped", "cc", "base", "armature", "rig",
           "ctrl", "ctl", "jnt", "bn", "j", "c"}
_PREFIX_WITH_DIGITS = {"mixamorig", "bip"}
_SIDE_L = {"left", "l"}
_SIDE_R = {"right", "r"}
_CENTRE = {"x", "m", "mid", "centre", "center"}
# Never map these: IK machinery, mechanism, helpers, correctives.
_HARD = {"ik", "pole", "target", "tgt", "pv", "vis", "mch", "tweak", "twist",
         "twst", "roll", "fix", "corrective", "correct", "helper", "master",
         "ref", "proxy", "aim", "offset", "pivot", "widget", "drv", "end",
         "tip", "nub", "stretch", "socket"}
_IK_LIKE = re.compile(r"^ik(target|pole|handle|foot|hand|knee|elbow|leg|arm|spine)?$|"
                      r".*(iktarget|ikpole|ikhandle)$")

_FINGER_RE = re.compile(r"^(?:hand)?(thumb|index|middle|ring|pinky|pinkie|little)(?:finger)?$")
_FINGER_CANON = {"pinkie": "pinky", "little": "pinky"}

# base word -> (canonical part, base score)
_BASE: Dict[str, Tuple[str, float]] = {}


def _reg(part, score, *words):
    for w in words:
        _BASE[w] = (part, score)


_reg("hips", 1.0, "hips", "pelvis", "roothips")
_reg("hips", 0.9, "hip")
_reg("spine*", 1.0, "spine", "chest", "upperchest", "lowerchest")
_reg("spine*", 0.9, "abdomen", "waist")
_reg("neck", 1.0, "neck")
_reg("head", 1.0, "head")
_reg("shoulder", 1.0, "shoulder", "clavicle", "collarbone")
_reg("shoulder", 0.9, "collar")
_reg("upper_arm", 1.0, "upperarm", "uparm", "humerus")
_reg("upper_arm", 0.9, "arm")
_reg("lower_arm", 1.0, "forearm", "lowerarm", "lowarm")
_reg("hand", 1.0, "hand")
_reg("hand", 0.7, "wrist")
_reg("upper_leg", 1.0, "thigh", "upleg", "upperleg", "femur")
_reg("lower_leg", 1.0, "shin", "calf", "lowerleg", "lowleg", "tibia")
_reg("lower_leg", 0.85, "leg")
_reg("foot", 1.0, "foot")
_reg("foot", 0.7, "ankle")
_reg("toes", 1.0, "toe", "toes", "toebase")
_reg("toes", 0.9, "ball")

_SPINE_RANK = {"spine": 0, "abdomen": 0, "waist": 0, "lowerchest": 1,
               "chest": 2, "upperchest": 3}


@dataclass
class ParsedName:
    side: Optional[str] = None      # 'l', 'r' or None
    base: str = ""
    nums: Tuple[int, ...] = ()
    flags: frozenset = frozenset()  # fk / def / org
    excluded: Optional[str] = None  # reason, or None
    dup: bool = False


def parse_name(name: str) -> ParsedName:
    p = ParsedName()
    if _DUP.search(name):
        p.dup = True
        name = _DUP.sub("", name)
    toks: List[str] = []
    for part in _SPLIT.split(name):
        toks += _CAMEL.findall(part)
    toks = [t.lower() for t in toks]
    flags = set()
    # peel leading prefixes
    while len(toks) > 1:
        t = toks[0]
        if t in ("def", "org"):
            flags.add(t)
            toks.pop(0)
        elif t in _PREFIX:
            toks.pop(0)
            if t in _PREFIX_WITH_DIGITS and toks and toks[0].isdigit():
                toks.pop(0)
        elif t == "f" and len(toks) > 1 and toks[1] in (
                "thumb", "index", "middle", "ring", "pinky"):
            toks.pop(0)  # Rigify finger prefix f_index.01.L
        else:
            break
    # side
    side = None
    rest: List[str] = []
    for i, t in enumerate(toks):
        if side is None and t in _SIDE_L:
            side = "l"
        elif side is None and t in _SIDE_R:
            side = "r"
        elif t in _CENTRE and i == len(toks) - 1 and i > 0:
            pass
        else:
            rest.append(t)
    p.side = side
    nums = tuple(int(t) for t in rest if t.isdigit())
    alpha = [t for t in rest if not t.isdigit()]
    for t in alpha:
        if t in ("fk", "def", "org"):
            flags.add(t)
        if t in _HARD or _IK_LIKE.match(t):
            p.excluded = t
    p.flags = frozenset(flags)
    p.nums = nums
    p.base = "".join(t for t in alpha if t not in ("fk", "def", "org"))
    if p.excluded is None and _IK_LIKE.match(p.base):
        p.excluded = p.base
    return p


def is_excluded_name(name: str) -> bool:
    """True for IK targets, pole bones, mechanism/tweak/twist/helper bones."""
    return parse_name(name).excluded is not None


# --------------------------------------------------------------------------
# Name pass
# --------------------------------------------------------------------------

_FLAG_MULT = {"fk": 1.0, "plain": 1.0, "def": 0.85, "org": 0.4}
_FK_BONUS = 0.05      # selection-only tie-break, not in the reported confidence
_TOPO_BONUS = 0.15    # selection-only
_TOPO_CONF = 0.05     # added to reported confidence when topology agrees
_DUP_MULT = 0.6
_DRIVEN_MULT = 0.6
_GEO_CONF = {"core": 0.6, "neck": 0.5, "finger": 0.45}


@dataclass
class _Cand:
    idx: int
    slot: str          # canonical name, or 'spine*'
    conf: float
    sel: float
    ident: tuple = ()
    z: float = 0.0
    nums: Tuple[int, ...] = ()
    base: str = ""
    note: str = ""


def _height(infos: Sequence[BoneInfo]) -> float:
    zs = [b.head[2] for b in infos] or [1.0]
    return max(max(zs) - min(zs), 1e-6)


def _is_ancestor(infos, anc: int, node: int, limit: int = 60) -> bool:
    p = infos[node].parent
    n = 0
    while p >= 0 and n < limit:
        if p == anc:
            return True
        p = infos[p].parent
        n += 1
    return False


def _name_candidates(infos: Sequence[BoneInfo]) -> List[_Cand]:
    H = _height(infos)
    eps = 0.01 * H
    cands: List[_Cand] = []
    for i, b in enumerate(infos):
        pn = parse_name(b.name)
        if pn.excluded is not None:
            continue
        part = None
        score = 1.0
        fm = _FINGER_RE.match(pn.base)
        if fm and pn.nums:
            n = pn.nums[-1]
            if 1 <= n <= 3:
                finger = _FINGER_CANON.get(fm.group(1), fm.group(1))
                part, score = f"{finger}_{n}", 1.0
        elif pn.base in _BASE:
            part, score = _BASE[pn.base]
        if part is None:
            continue
        # rig flavour multiplier
        if "fk" in pn.flags:
            mult, bonus = _FLAG_MULT["fk"], _FK_BONUS
        elif "def" in pn.flags:
            mult, bonus = _FLAG_MULT["def"], 0.0
        elif "org" in pn.flags:
            mult, bonus = _FLAG_MULT["org"], 0.0
        else:
            mult, bonus = _FLAG_MULT["plain"], 0.0
        note = []
        if pn.dup:
            mult *= _DUP_MULT
            note.append("duplicate suffix")
        if b.driven:
            mult *= _DRIVEN_MULT
            note.append("driven by constraint/driver")
        sided = part not in ("hips", "spine*", "neck", "head")
        side = pn.side
        x = b.head[0]
        if sided:
            if side is None:
                if abs(x) > eps:
                    side = "l" if x > 0 else "r"
                    mult *= 0.6
                    note.append("side from x position")
                else:
                    continue
            elif abs(x) > eps and (x > 0) != (side == "l"):
                mult *= 0.5
                note.append("name side disagrees with x position")
        slot = part if part == "spine*" or not sided else f"{part}_{side}"
        if slot != "spine*" and slot not in C.BY_NAME:
            continue
        conf = min(1.0, score * mult)
        cands.append(_Cand(i, slot, conf, conf + bonus, ident=(pn.base, pn.nums, side),
                           z=b.head[2], nums=pn.nums, base=pn.base,
                           note="; ".join(note)))
    return cands


def _name_pass(infos: Sequence[BoneInfo]) -> Dict[str, Tuple[int, float, str]]:
    cands = _name_candidates(infos)
    by_slot: Dict[str, List[_Cand]] = {}
    for c in cands:
        by_slot.setdefault(c.slot, []).append(c)

    def best(slot, cl):
        if slot == "neck":
            top = max(c.sel for c in cl)
            near = [c for c in cl if c.sel >= top - 0.1]
            return min(near, key=lambda c: (c.nums, c.z, infos[c.idx].name))
        return max(cl, key=lambda c: (round(c.sel, 4), -len(infos[c.idx].name), infos[c.idx].name))

    # pass 1: provisional picks
    pick: Dict[str, _Cand] = {s: best(s, cl) for s, cl in by_slot.items() if s != "spine*"}

    # pass 2: topology bonus against neighbours' provisional picks
    pairs = [(b.parent, b.name) for b in C.BONES
             if b.parent is not None and b.group not in ("spine",)
             and b.parent not in ("upper_chest",)]
    children_of: Dict[str, List[str]] = {}
    for par, ch in pairs:
        children_of.setdefault(par, []).append(ch)
    consistent: Dict[str, bool] = {}
    final: Dict[str, _Cand] = {}
    for slot, cl in by_slot.items():
        if slot == "spine*":
            continue
        par = C.BY_NAME[slot].parent
        scored = []
        for c in cl:
            bonus = 0.0
            ok = False
            pp = pick.get(par) if par else None
            if pp is not None and pp.idx != c.idx and _is_ancestor(infos, pp.idx, c.idx):
                ok = True
            for ch in children_of.get(slot, ()):
                cp = pick.get(ch)
                if cp is not None and cp.idx != c.idx and _is_ancestor(infos, c.idx, cp.idx):
                    ok = True
            if ok:
                bonus = _TOPO_BONUS
            scored.append((c.sel + bonus, c, ok))
        if slot == "neck":
            top = max(s for s, _, _ in scored)
            near = [t for t in scored if t[0] >= top - 0.1]
            s, c, ok = min(near, key=lambda t: (t[1].nums, t[1].z, infos[t[1].idx].name))
        else:
            s, c, ok = max(scored, key=lambda t: (round(t[0], 4), -len(infos[t[1].idx].name), infos[t[1].idx].name))
        final[slot] = c
        consistent[slot] = ok

    result: Dict[str, Tuple[int, float, str]] = {}
    used = set()
    for slot, c in final.items():
        conf = min(1.0, c.conf + (_TOPO_CONF if consistent[slot] else 0.0))
        result[slot] = (c.idx, conf, c.note)
        used.add(c.idx)

    # spine chain
    sp = [c for c in by_slot.get("spine*", []) if c.idx not in used]
    if sp:
        # keep the best candidate per identity (rid of FK/DEF/ORG duplicates)
        per_ident: Dict[tuple, _Cand] = {}
        for c in sp:
            cur = per_ident.get(c.ident)
            if cur is None or c.sel > cur.sel:
                per_ident[c.ident] = c
        members = list(per_ident.values())
        fam: Dict[str, int] = {}
        for c in members:
            fam[c.base] = fam.get(c.base, 0) + 1
        big = [b for b, n in fam.items() if n >= 3]
        if big:  # a numbered chain (spine_01..05) wins over stray 'chest' controls
            fb = max(big, key=lambda b: fam[b])
            members = [c for c in members if c.base == fb]
        members.sort(key=lambda c: (_SPINE_RANK.get(c.base, 0), c.nums, c.z))
        n = len(members)
        if n == 1:
            slots = [(0, "spine")]
        elif n == 2:
            slots = [(0, "spine"), (1, "chest")]
        else:
            slots = [(0, "spine"), (n // 2, "chest"), (n - 1, "upper_chest")]
        for k, name in slots:
            c = members[k]
            result[name] = (c.idx, c.conf, c.note)
    return result


# --------------------------------------------------------------------------
# Geometry / topology fallback
# --------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _len(a):
    return math.sqrt(_dot(a, a))


def _norm(a):
    l = _len(a) or 1.0
    return (a[0] / l, a[1] / l, a[2] / l)


class _Tree:
    """Hierarchy over the allowed (non-excluded) bones with subtree stats."""

    def __init__(self, infos: Sequence[BoneInfo]):
        self.infos = infos
        allowed = [i for i, b in enumerate(infos) if not is_excluded_name(b.name)]
        deform = [i for i in allowed if infos[i].deform]
        if len(deform) >= 15:
            allowed = deform
        aset = set(allowed)
        self.parent: Dict[int, int] = {}
        self.children: Dict[int, List[int]] = {i: [] for i in allowed}
        for i in allowed:
            p = infos[i].parent
            while p >= 0 and p not in aset:
                p = infos[p].parent
            self.parent[i] = p
            if p >= 0:
                self.children[p].append(i)
        self.roots = [i for i in allowed if self.parent[i] < 0]
        # post-order stats
        self.h: Dict[int, int] = {}
        self.minx: Dict[int, float] = {}
        self.maxx: Dict[int, float] = {}
        self.minz: Dict[int, float] = {}
        self.maxz: Dict[int, float] = {}
        self.depth: Dict[int, int] = {}
        order: List[int] = []
        stack = [(r, 0) for r in self.roots]
        while stack:
            n, d = stack.pop()
            self.depth[n] = d
            order.append(n)
            for c in self.children[n]:
                stack.append((c, d + 1))
        for n in reversed(order):
            hd = infos[n].head
            h, mnx, mxx, mnz, mxz = 0, hd[0], hd[0], hd[2], hd[2]
            for c in self.children[n]:
                h = max(h, self.h[c] + 1)
                mnx = min(mnx, self.minx[c])
                mxx = max(mxx, self.maxx[c])
                mnz = min(mnz, self.minz[c])
                mxz = max(mxz, self.maxz[c])
            self.h[n], self.minx[n], self.maxx[n], self.minz[n], self.maxz[n] = h, mnx, mxx, mnz, mxz
        H = max(self.maxz.values(), default=1.0) - min(self.minz.values(), default=0.0)
        self.H = max(H, 1e-6)

    def length(self, i):
        b = self.infos[i]
        return _len(_sub(b.tail, b.head))

    def drop_nub(self, path: List[int]) -> List[int]:
        """Remove trailing end/nub bones (much shorter than their parent)."""
        while len(path) >= 2 and self.length(path[-1]) < 0.45 * self.length(path[-2]):
            path = path[:-1]
        return path

    def is_lateral(self, c: int, node: int) -> int:
        """+1 / -1 if the subtree of c extends far to the character's left / right."""
        x = self.infos[node].head[0]
        pos, neg = self.maxx[c] - x, x - self.minx[c]
        thr = 0.12 * self.H
        if pos > thr and pos > 2 * neg:
            return 1
        if neg > thr and neg > 2 * pos:
            return -1
        return 0

    def walk(self, start: int, key) -> List[int]:
        path = [start]
        while self.children[path[-1]]:
            path.append(max(self.children[path[-1]], key=key))
        return path


def _geometry_pass(infos: Sequence[BoneInfo]) -> Dict[str, Tuple[int, float, str]]:
    T = _Tree(infos)
    H = T.H
    out: Dict[str, Tuple[int, float, str]] = {}
    if not T.children:
        return out

    def put(slot, idx, conf, note="geometry"):
        out[slot] = (idx, conf, note)

    # --- hips: node with an upward branch and two downward branches (L and R)
    hips = None
    for n in sorted(T.children, key=lambda n: T.depth[n]):
        kids = T.children[n]
        if len(kids) < 3:
            continue
        nz = infos[n].head[2]
        ups = [c for c in kids if T.maxz[c] > nz + 0.2 * H]
        downs = [c for c in kids if T.minz[c] < nz - 0.2 * H and T.maxz[c] <= nz + 0.05 * H]
        xs = [infos[c].head[0] for c in downs]
        if ups and any(x > 0.01 * H for x in xs) and any(x < -0.01 * H for x in xs):
            hips = (n, ups, downs)
            break
    if hips is None:
        return out
    hn, ups, downs = hips
    put("hips", hn, _GEO_CONF["core"])

    # --- legs
    for side, sign in (("l", 1), ("r", -1)):
        sd = [c for c in downs if sign * infos[c].head[0] > 0.01 * H]
        if not sd:
            continue
        t = max(sd, key=lambda c: (T.h[c], -T.minz[c]))
        path = T.walk(t, key=lambda c: (-T.minz[c], T.h[c]))
        path = T.drop_nub(path)
        for k, part in enumerate(("upper_leg", "lower_leg", "foot", "toes")):
            if k < len(path):
                put(f"{part}_{side}", path[k], _GEO_CONF["core"])

    # --- spine chain up to the arm branch
    start = max(ups, key=lambda c: T.maxz[c])
    chain = [start]
    branch = None
    while True:
        node = chain[-1]
        kids = T.children[node]
        lat = {c: T.is_lateral(c, node) for c in kids}
        if any(v > 0 for v in lat.values()) and any(v < 0 for v in lat.values()):
            branch = node
            break
        mid = [c for c in kids if lat[c] == 0] or kids
        if not mid:
            break
        chain.append(max(mid, key=lambda c: (T.maxz[c], T.h[c])))
    n = len(chain)
    spine_slots = ([(0, "spine")] if n == 1 else [(0, "spine"), (1, "chest")] if n == 2
                   else [(0, "spine"), (n // 2, "chest"), (n - 1, "upper_chest")])
    for k, name in spine_slots:
        put(name, chain[k], _GEO_CONF["core"])
    if branch is None:
        return out

    kids = T.children[branch]
    lat = {c: T.is_lateral(c, branch) for c in kids}

    # --- neck + head
    mid = [c for c in kids if lat[c] == 0]
    if mid:
        top = max(mid, key=lambda c: T.maxz[c])
        path = T.drop_nub(T.walk(top, key=lambda c: (T.maxz[c], T.h[c])))
        hi = len(path) - 1
        for k, nd in enumerate(path):
            if k > 0 and len(T.children[nd]) >= 2:
                hi = k
                break
        put("head", path[hi], _GEO_CONF["core"])
        if hi > 0:
            put("neck", path[0], _GEO_CONF["neck"])

    # --- arms and fingers
    for side, sign in (("l", 1), ("r", -1)):
        sa = [c for c in kids if lat[c] == sign]
        if not sa:
            continue
        a0 = max(sa, key=lambda c: sign * (T.maxx[c] if sign > 0 else T.minx[c]))
        path = T.walk(a0, key=lambda c: sign * (T.maxx[c] if sign > 0 else T.minx[c]))
        hand_k = None
        for k, nd in enumerate(path):
            if sum(1 for c in T.children[nd] if T.h[c] >= 1) >= 3:
                hand_k = k
                break
        if hand_k is None:
            path = T.drop_nub(path)
            hand_k = min(len(path) - 1, 3)
        for off, part in ((0, "hand"), (1, "lower_arm"), (2, "upper_arm"), (3, "shoulder")):
            if hand_k - off >= 0:
                put(f"{part}_{side}", path[hand_k - off], _GEO_CONF["core"])
        _geometry_fingers(T, path[hand_k], side, put)
    return out


def _geometry_fingers(T: _Tree, hand: int, side: str, put):
    infos = T.infos
    tips = []
    stack = list(T.children[hand])
    while stack:
        n = stack.pop()
        if T.children[n]:
            stack.extend(T.children[n])
        else:
            tips.append(n)
    chains = []
    for tip in tips:
        path = []
        n = tip
        while n != hand and n >= 0:
            path.append(n)
            n = T.parent[n]
        path.reverse()
        path = T.drop_nub(path)
        if len(path) >= 3:
            chains.append(path)
    if len(chains) < 5:
        return
    hp = infos[hand].head
    chains.sort(key=lambda p: -_len(_sub(infos[p[-1]].head, hp)))
    chains = chains[:5]
    tipsp = [infos[p[-1]].head for p in chains]
    mean = tuple(sum(t[k] for t in tipsp) / 5 for k in range(3))
    f = _norm(_sub(mean, hp))
    offs = []
    for t in tipsp:
        d = _sub(t, mean)
        dp = _dot(d, f)
        offs.append((d[0] - dp * f[0], d[1] - dp * f[1], d[2] - dp * f[2]))
    # principal axis of the offsets by power iteration
    v = (1.0, 0.31, 0.17)
    for _ in range(30):
        w = [0.0, 0.0, 0.0]
        for o in offs:
            s = _dot(o, v)
            for k in range(3):
                w[k] += o[k] * s
        v = _norm(tuple(w)) if _len(w) > 1e-12 else v
    order = sorted(range(5), key=lambda i: _dot(offs[i], v))
    proj = [_dot(offs[i], v) for i in order]
    thumb_first = (proj[1] - proj[0]) > (proj[4] - proj[3])
    seq = order if thumb_first else order[::-1]
    for finger, ci in zip(C.FINGERS, seq):
        last3 = chains[ci][-3:]
        for k, nd in enumerate(last3):
            put(f"{finger}_{k + 1}_{side}", nd, _GEO_CONF["finger"])


# --------------------------------------------------------------------------
# Public mapping API
# --------------------------------------------------------------------------

def map_bones(infos: Sequence[BoneInfo], overrides: Optional[Dict[str, str]] = None,
              armature_name: str = "", use_geometry: bool = True) -> RigMapping:
    """Map `infos` onto the canonical skeleton. Pure Python."""
    mapping = RigMapping(armature=armature_name)
    index = {b.name: i for i, b in enumerate(infos)}
    used = set()

    for slot, (idx, conf, note) in _name_pass(infos).items():
        mapping.bones[slot] = BoneMatch(infos[idx].name, conf, "name", note)
        used.add(idx)

    if use_geometry and mapping.unmapped:
        geo = _geometry_pass(infos)
        # If geometry and names disagree about the hips, the rig is not what
        # the geometry assumes (or names are right and geometry is confused):
        # trust names and fill nothing.
        gh = geo.get("hips")
        if gh is not None and "hips" in mapping.bones and infos[gh[0]].name != mapping.bone("hips"):
            geo = {}
        for slot, (idx, conf, note) in geo.items():
            if slot in mapping.bones or idx in used:
                continue
            mapping.bones[slot] = BoneMatch(infos[idx].name, conf, "geometry", note)
            used.add(idx)

    for slot, bone in (overrides or {}).items():
        if slot in C.BY_NAME and bone in index:
            mapping.bones[slot] = BoneMatch(bone, 1.0, "manual")
    return mapping


_DRIVING_CONSTRAINTS = {"COPY_TRANSFORMS", "COPY_ROTATION", "ARMATURE", "TRACK_TO",
                        "DAMPED_TRACK", "LOCKED_TRACK", "IK", "SPLINE_IK", "TRANSFORMATION"}


def bone_infos(obj) -> List[BoneInfo]:
    """Extract rest-pose data (world space) from an armature object."""
    arm = obj.data
    mw = obj.matrix_world
    bones = list(arm.bones)
    index = {b.name: i for i, b in enumerate(bones)}
    driven_names = set()
    ad = obj.animation_data
    if ad is not None:
        for fc in ad.drivers:
            dp = fc.data_path
            if dp.startswith('pose.bones["') and ".rotation" in dp:
                driven_names.add(dp[len('pose.bones["'):dp.index('"]')])
    pose_bones = obj.pose.bones if obj.pose else {}
    infos = []
    for b in bones:
        h = mw @ b.head_local
        t = mw @ b.tail_local
        driven = b.name in driven_names
        pb = pose_bones.get(b.name) if obj.pose else None
        if pb is not None and not driven:
            for con in pb.constraints:
                if (con.type in _DRIVING_CONSTRAINTS and not con.mute
                        and con.influence > 0.99):
                    driven = True
                    break
        infos.append(BoneInfo(b.name, index[b.parent.name] if b.parent else -1,
                              (h.x, h.y, h.z), (t.x, t.y, t.z), b.use_deform, driven))
    return infos


def map_armature(obj, overrides: Optional[Dict[str, str]] = None,
                 use_geometry: bool = True) -> RigMapping:
    """Auto-map armature object `obj`. Never renames or modifies it."""
    if obj is None or obj.type != 'ARMATURE':
        raise TypeError("map_armature needs an armature object")
    return map_bones(bone_infos(obj), overrides, obj.name, use_geometry)


def format_report(mapping: RigMapping) -> str:
    """Human-readable mapping table."""
    lines = [f"armature: {mapping.armature}  required confidence: {mapping.confidence:.3f}  "
             f"mapped {len(mapping.bones)}/{len(C.NAMES)}"]
    for n, b in mapping.mapped_items():
        m = mapping.bones[n]
        lines.append(f"  {n:<16} -> {b:<28} {m.confidence:.2f} {m.method}"
                     + (f"  ({m.note})" if m.note else ""))
    if mapping.unmapped:
        lines.append("  unmapped: " + ", ".join(mapping.unmapped))
    if mapping.required_unmapped:
        lines.append("  REQUIRED unmapped: " + ", ".join(mapping.required_unmapped))
    return "\n".join(lines)
