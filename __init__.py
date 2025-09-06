# __init__.py
# Wrapper/loader for the Section Box + Planar Sections add-on.
# - Delegates register()/unregister() to your core module.
# - Supports multiple possible core filenames so you can rename without touching __init__.

bl_info = {
    "name": "Section Toolbox",
    "author": "Victor Calixto",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
    "location": "3D Viewport > N-panel > Section Box",
    "description": "Parametric section box and planar cuts with per-face outputs and SVG/DXF export",
    "category": "3D View",
}

import importlib
import sys

# Try these modules (relative to this package) in order.
# Put your main implementation in one of these files.
_CORE_CANDIDATES = (
    ".section_tool4",          # preferred current name
    ".section_tool",           # older name
    ".section_box_planar",     # alternate
    ".section_box",            # legacy
)

_CORE = None
_LAST_ERR = None


def _load_core():
    """Import or reload the first available core module from _CORE_CANDIDATES."""
    global _CORE, _LAST_ERR
    pkg = __package__ or __name__  # package name when installed as a module

    # Reload if we already have one (useful when re-registering in dev)
    if _CORE is not None:
        try:
            _CORE = importlib.reload(_CORE)
            return _CORE
        except Exception as e:
            _LAST_ERR = e
            _CORE = None  # fall through to clean import attempts

    # Fresh import attempts in order
    for relname in _CORE_CANDIDATES:
        try:
            _CORE = importlib.import_module(relname, pkg)
            return _CORE
        except Exception as e:
            _LAST_ERR = e
            continue

    # Nothing worked
    names = ", ".join(n.lstrip(".") for n in _CORE_CANDIDATES)
    raise ImportError(
        f"[SectionBox] Could not import a core module. "
        f"Tried: {names}. Last error: {_LAST_ERR}"
    ) from _LAST_ERR


def register():
    core = _load_core()
    if not hasattr(core, "register"):
        raise AttributeError(
            "[SectionBox] Core module is missing register(). "
            "Ensure your core file defines register()/unregister()."
        )
    core.register()


def unregister():
    global _CORE
    if _CORE and hasattr(_CORE, "unregister"):
        try:
            _CORE.unregister()
        except Exception:
            # Don't block Blender's shutdown/uninstall on errors here.
            pass
    _CORE = None


# Allow running this file directly in Blender's text editor during development.
if __name__ == "__main__":
    register()

