# Section Toolbox (STB)

**Section Toolbox** adds parametric **section boxes** and **planar sections** to Blender with live controls, versioned outputs, and clean 2D exports (SVG/DXF). Drive per-axis anchors, live move/rotate/resize, face-based **cut** & **projection** lines, **filled cut faces**, and planar **elevations** with direction + depth. Elevations/projections are **occlusion-aware** (near geometry hides far geometry).

![Icon](icon.png)

---

## Features

### Section Box
- One-click box that **fits the current selection** (evaluated bounds; modifiers included).
- **Anchors** per axis (Center / − / +) to pin faces while resizing.
- **Live** Width / Depth / Height, **Move**, and **Rotate** (no apply step).
- Face outputs:
  - `_cut` — exact intersection, clipped to the face rectangle.
  - `_projection` — orthographic edges inside the box slab with **hidden-line removal**.
- **Filled cut faces** (for closed loops).
- Each run is **versioned** and organised in nested collections.

### Planar Sections
- Create cutting planes in **XY / XZ / YZ**.
- Set **Direction** (+N/−N) and **Depth** to define the elevation slab.
- **Live** plane sizing (**Size U/V**) with anchors (Center / −U / +U / −V / +V).
- Stores a **frozen plane copy** per run.
- Elevation lines are **occlusion-aware** within the slab.

### Export
- **SVG**: thin strokes (0.25) + **filled polygons** for closed cuts.
- **DXF**: LINE entities at **Z=0** (pure 2D).
- Orientation comes from a saved **Frame2D** so exports are **view-aligned**.
- **Flip X / Flip Y** toggles to match your drawing convention.

> Note: Internally, generated data may use `SBX_` / `SBP_` prefixes for compatibility. This does not affect usage.

---

## Quick Start

### Box Workflow
1. Select the meshes to section → **Create Box** (fits selection).
2. Adjust **Width/Depth/Height** with anchors; **Move/Rotate** live.
3. Pick a face (±X / ±Y / ±Z) → **Generate Box Section**.
4. **Export** → _Export SVG (last Box run)_ or _Export DXF (last Box run)_.  
If the 2D looks mirrored/flipped, toggle **Flip X / Flip Y**.

### Planar Workflow
1. Choose **Plane** (XY/XZ/YZ), **Direction** (+N/−N), and **Depth**.
2. **Create Plane** → resize live (Size U/V + anchors).
3. Bind selection (or rely on current selection) → **Generate Planar Section**.
4. **Export** → _Export SVG/DXF (last Plane run)_.

---

## Collections Layout

- **Box run**

STB_Results_<BoxName>/Section_###_<Face>/
├─ __Cut/ (…Line_cut…)
├─ __Meshes/ (…FillFace…)
└─ __Projection/ (…Line_projection…)

 **Plane run**
 
TB_Results_<PlaneName>/Section_###_<Tag>/
├─ __Plane/ (frozen plane copy)
├─ __Cut/ (…Line_cut…)
├─ __Meshes/ (…FillFace…)
└─ __Elevation/ (…Line_elev…)


> Depending on the build, internal prefixes may appear as `SBX_Results_…` / `SBP_Results_…`.

---

## Export Notes
- **SVG:** stroke width 0.25; closed cuts export as filled polygons; suitable for CAD and illustration.
- **DXF:** pure LINEs at Z=0; open in CAD with no extra flattening.
- Exporters always target the **most recent** run for the active Box/Plane.

---

## Troubleshooting
- **Hidden lines still visible:** Increase **Visibility Samples**; keep **HLR Epsilon** small (raise slightly only to avoid self-hits).
- **Nothing exported:** You must **Generate** first; exporters use the *last run* for the active object.
- **Mirrored/rotated 2D:** Use **Flip X / Flip Y** (applies to SVG & DXF).
- **Property errors:** Blender cannot register properties starting with `_`; ensure public names.
- **Performance:** Disable **Live** booleans for heavy scenes; re-enable before generating, or **Bake / Apply** when final.

---

## License
**GPL-3.0**
