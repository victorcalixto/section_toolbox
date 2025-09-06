# Section Toolbox (STB)

**Section Toolbox** brings parametric **section boxes** and **planar sections** to Blender with live controls, versioned results, and clean 2D exports (SVG/DXF). It includes per‑axis anchors, live move/rotate/resize, face‑based cut & projection lines, filled cut faces, and planar elevations with direction + depth.

<img alt="Icon" src="icon.png" width="96" />

## Features

- **Section Box**
  - Create a box that **fits the current selection** (evaluated bounds, modifiers included).
  - **Anchors** per axis (Center / −X / +X, etc.) so growth can pin faces.
  - **Live** Width/Depth/Height, Move, and Rotate (no “apply” button needed).
  - Face‑based outputs: `_cut` (true intersection), `_projection` (slab‑clipped), and **filled cut faces**.
  - Versioned results in nested collections per run.

- **Planar Sections**
  - Create cutting planes **XY / XZ / YZ**.
  - Set **Direction** (+N/−N) and **Depth** to generate elevation slabs.
  - **Live** plane sizing (Size U/V) with anchors (Center / −U / +U, etc.).
  - Versioned results in nested collections per run, with a frozen copy of the plane.

- **Export**
  - **SVG**: 2D export with thin strokes (0.25) and **filled polygons** for closed cuts.
  - **DXF**: line entities (Z=0).

> Note: Internally the tool may still name generated objects with `SBX_`/`SBP_` prefixes for compatibility. That does not affect usage.

## Install

### Classic Add-on (.zip)
1. Create this folder structure and zip the **folder** (not just the files):
```
section_toolbox_ext/
├─ __init__.py
├─ section_tool.py
├─ README.md
├─ LICENSE
└─ icon.png
```
2. Blender → **Edit → Preferences → Add-ons → Install…** → select the zip → Enable **Section Toolbox**.

### Blender Extensions (Blender 4.2+)
- Add a `manifest.yaml` at repository root per Extensions spec.
- Publish the repo publicly and submit it to the Extensions platform.

## Quick Start

### Box Workflow
1. Select the objects to section → **Create Box** (fits selection).
2. Adjust **Width/Depth/Height** with anchors; **Move/Rotate** live.
3. Choose a face (±X/±Y/±Z) → **Generate Box Section**.
4. Export with **Export SVG (last Box run)** or **Export DXF (last Box run)**.

### Planar Workflow
1. Choose **Plane** (XY/XZ/YZ), **Direction** (+N/−N) and **Depth**.
2. **Create Plane** and resize live (Size U/V + anchors).
3. Bind selection (or use current selection) → **Generate Planar Section**.
4. Export with **Export SVG (last Plane run)** or **Export DXF (last Plane run)**.

## Collections Layout

- **Box run**
  - `STB_Results_<BoxName>/Section_###_<Face>/`
    - `__Cut/` (…_Line_cut_…)
    - `__Meshes/` (…_FillFace_…)
    - `__Projection/` (…_Line_projection_…)

- **Plane run**
  - `STB_Results_<PlaneName>/Section_###_<Tag>/`
    - `__Plane/` (frozen copy of the plane)
    - `__Cut/` (…_Line_cut_…)
    - `__Meshes/` (…_FillFace_…)
    - `__Elevation/` (…_Line_elev_…)

## Export Notes

- **SVG**: thin stroke (0.25), fills for closed cuts; suitable for CAD and illustration.
- **DXF**: LINE entities; coordinates projected to 2D (Z=0).

## Troubleshooting

- **Property registration errors**: Blender won’t register properties starting with “_”. Rename them without underscores.
- **DXF/SVG issues**: The exporter writes line‑by‑line (DXF) and uses XML builders (SVG) to avoid quoting and EOL mistakes.
- **Objects outside collections**: The add‑on links generated objects **exclusively** to the results collection.

## License

GPL‑3.0‑only — see `LICENSE`.

## Credits

© 2025 Victor Calixto. Logo & docs adapted for **Section Toolbox (STB)**.
