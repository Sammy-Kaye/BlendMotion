"""BlendMotion (package name `handle`): direct-manipulation posing for humanoid rigs."""

bl_info = {
    "name": "BlendMotion",
    "author": "Samuel",
    "version": (0, 0, 1),
    "blender": (5, 0, 0),
    "location": "3D Viewport > N Panel > BlendMotion",
    "description": "Drag a hand, foot, hips or head and the whole body follows",
    "category": "Animation",
}


def register():
    from . import ui
    ui.register()


def unregister():
    from . import ui
    ui.unregister()
