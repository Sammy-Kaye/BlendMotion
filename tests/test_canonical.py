from handle.rig import canonical as C


def test_counts_and_parents():
    assert len(C.NAMES) == 52, len(C.NAMES)
    assert len(set(C.NAMES)) == len(C.NAMES)
    for b in C.BONES:
        assert b.parent is None or b.parent in C.BY_NAME, b
    assert [b.name for b in C.BONES if b.parent is None] == ["hips"]
    for n in ("hips", "spine", "head", "upper_arm_l", "hand_r", "upper_leg_l", "foot_r"):
        assert n in C.REQUIRED
    for n in ("toes_l", "thumb_1_l", "chest", "shoulder_r"):
        assert n not in C.REQUIRED


def test_sides_and_mirror():
    for b in C.BONES:
        if b.side:
            assert b.name.endswith("_" + b.side)
            assert C.mirror(b.name) in C.BY_NAME
            assert C.mirror(C.mirror(b.name)) == b.name
    assert C.mirror("head") == "head"
    assert C.mirror("index_2_l") == "index_2_r"


def test_resolved_parent_skips_missing_optional():
    present = set(C.NAMES) - {"chest", "upper_chest", "shoulder_l"}
    assert C.resolved_parent("neck", present) == "spine"
    assert C.resolved_parent("upper_arm_l", present) == "spine"
    assert C.resolved_parent("hips", present) is None
