import os
import tomllib

import bpy

import rigs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_manifest():
    with open(os.path.join(ROOT, "handle", "blender_manifest.toml"), "rb") as f:
        d = tomllib.load(f)
    assert d["id"] == "handle" and d["type"] == "add-on"
    assert d["blender_version_min"].startswith("5.")
    assert len(d["tagline"]) <= 64 and not d["tagline"].endswith(".")
    assert d["schema_version"] == "1.0.0"


def test_register_panel_and_operators():
    rigs.reset()
    import handle
    handle.register()
    try:
        assert hasattr(bpy.types, "HANDLE_PT_main")
        assert bpy.types.HANDLE_PT_main.bl_category == "BlendMotion"
        assert bpy.types.HANDLE_PT_main.bl_space_type == 'VIEW_3D'
        obj = rigs.build_mixamo()
        bpy.context.view_layer.objects.active = obj
        assert bpy.ops.handle.detect_rig() == {'FINISHED'}
        from handle.rig import store
        m = store.fetch_mapping(obj)
        assert m is not None and m.is_usable and m.confidence > 0.95
        assert bpy.ops.handle.key_pose() == {'FINISHED'}
        assert obj.animation_data and obj.animation_data.action
        assert bpy.ops.handle.clear_mapping() == {'FINISHED'}
        assert store.fetch_mapping(obj) is None
    finally:
        handle.unregister()
    assert not hasattr(bpy.types, "HANDLE_PT_main")
