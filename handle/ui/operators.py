import bpy

from ..rig import mapper, pose_io, store


def _active_armature(context):
    obj = context.active_object
    if obj is not None and obj.type == 'ARMATURE':
        return obj
    for o in context.selected_objects:
        if o.type == 'ARMATURE':
            return o
    return None


class HANDLE_OT_detect_rig(bpy.types.Operator):
    bl_idname = "handle.detect_rig"
    bl_label = "Detect Rig"
    bl_description = "Map the active armature's bones to the canonical humanoid (no renaming)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def execute(self, context):
        obj = _active_armature(context)
        old = store.fetch_mapping(obj)
        overrides = ({k: v.bone for k, v in old.bones.items() if v.method == "manual"}
                     if old else None)
        m = mapper.map_armature(obj, overrides)
        store.store_mapping(obj, m)
        self.report({'INFO'}, f"Mapped {len(m.bones)} bones, confidence {m.confidence:.2f}")
        return {'FINISHED'}


class HANDLE_OT_clear_mapping(bpy.types.Operator):
    bl_idname = "handle.clear_mapping"
    bl_label = "Clear Mapping"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = _active_armature(context)
        return obj is not None and store.fetch_mapping(obj) is not None

    def execute(self, context):
        store.clear_mapping(_active_armature(context))
        return {'FINISHED'}


class HANDLE_OT_key_pose(bpy.types.Operator):
    bl_idname = "handle.key_pose"
    bl_label = "Key Current Pose"
    bl_description = "Insert FK rotation keyframes for all mapped bones at the current frame"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = _active_armature(context)
        return obj is not None and store.fetch_mapping(obj) is not None

    def execute(self, context):
        obj = _active_armature(context)
        m = store.fetch_mapping(obj)
        snap = pose_io.read_pose(obj, m)
        n = pose_io.write_pose(obj, m, snap.rotations, undo_label=None)  # operator owns the undo step
        self.report({'INFO'}, f"Keyed {n} bones")
        return {'FINISHED'}


classes = (HANDLE_OT_detect_rig, HANDLE_OT_clear_mapping, HANDLE_OT_key_pose)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
