# __init__.py — Section Toolbox (Extension entry)
# Minimal, reviewer-friendly wrapper.

bl_info = {
    "name": "Section Toolbox",
    "author": "Victor Calixto",
    "version": (0, 0, 2),
    "blender": (4, 2, 0),
    "location": "3D Viewport > N-panel > Section Toolbox",
    "description": "Parametric section boxes and planar cuts with live controls, filled cuts, and clean SVG/DXF 2D exports.",
    "category": "Object",
}

from . import section_tool as _core

def register():
    _core.register()

def unregister():
    _core.unregister()

if __name__ == "__main__":
    register()

