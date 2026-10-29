import bpy

from ..rig import canonical, store
from .operators import _active_armature


class HANDLE_PT_main(bpy.types.Panel):
    bl_label = "BlendMotion"
    bl_idname = "HANDLE_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "BlendMotion"

    def draw(self, context):
        layout = self.layout
        obj = _active_armature(context)
        if obj is None:
            layout.label(text="Select an armature", icon='ARMATURE_DATA')
            return
        layout.label(text=obj.name, icon='ARMATURE_DATA')
        m = store.fetch_mapping(obj)
        if m is None:
            layout.label(text="Rig not detected yet", icon='QUESTION')
            layout.operator("handle.detect_rig", icon='VIEWZOOM')
            return

        col = layout.column(align=True)
        total = len(canonical.NAMES)
        col.label(text=f"Mapped {len(m.bones)} / {total} bones")
        col.label(text=f"Confidence {m.confidence * 100:.0f}%",
                  icon='CHECKMARK' if m.is_usable and m.confidence >= 0.8 else 'ERROR')
        req = m.required_unmapped
        if req:
            box = layout.box()
            box.label(text="Missing required bones:", icon='ERROR')
            for n in req[:8]:
                box.label(text=n)
            if len(req) > 8:
                box.label(text=f"... and {len(req) - 8} more")
        else:
            layout.label(text="All required bones mapped", icon='CHECKMARK')
        low = [n for n, b in m.bones.items() if b.confidence < 0.6]
        if low:
            layout.label(text=f"{len(low)} low-confidence bones", icon='INFO')
        row = layout.row(align=True)
        row.operator("handle.detect_rig", text="Re-detect", icon='FILE_REFRESH')
        row.operator("handle.clear_mapping", text="", icon='X')
        layout.operator("handle.key_pose", icon='KEY_HLT')


def register():
    bpy.utils.register_class(HANDLE_PT_main)


def unregister():
    bpy.utils.unregister_class(HANDLE_PT_main)
