# Section Toolbox — Blender Add-on (GPL v3)
# Author: Victor Calixto
# Version: 0.0.1
# Blender: 3.5–4.5+

bl_info = {
    "name": "Section Toolbox",
    "author": "Victor Calixto",
    "version": (0, 0, 1),
    "blender": (3, 5, 0),
    "location": "3D Viewport > N-panel > Section Box",
    "description": "Box & planar sectioning with live params, anchors, filled cuts, versioned collections, and SVG/DXF export",
    "category": "3D View",
}

import bpy
import bmesh
import os
from mathutils import Vector, Matrix, Euler
from mathutils.bvhtree import BVHTree
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    BoolProperty, StringProperty, EnumProperty, PointerProperty, FloatProperty, IntProperty
)
import xml.etree.ElementTree as ET

# ------------------------------- Constants ----------------------------------

BOX_PREFIX    = "SectionBox"
PLANE_PREFIX  = "SectionPlane"
MOD_PREFIX    = "SBX_"
PROP_TARGETS  = "sbx_targets"
PROP_PTARGETS = "sbp_targets"
PROP_LAST_ITER_BOX  = "sbx_last_iter_name"
PROP_LAST_ITER_TAGB = "sbx_last_iter_tag"
PROP_LAST_ITER_PLN  = "sbp_last_iter_name"
PROP_LAST_ITER_TAGP = "sbp_last_iter_tag"

# -------------------------------- Utilities ---------------------------------

def _get_active_box(context):
    ob = context.active_object
    if ob and ob.type == "MESH" and ob.name.startswith(BOX_PREFIX):
        return ob
    return None

def _get_active_plane(context):
    ob = context.active_object
    if ob and ob.type == "MESH" and ob.name.startswith(PLANE_PREFIX):
        return ob
    return None

def _objects_from_target_mode(context, mode="SELECTION", collection_name=""):
    objs = []
    if mode == "SELECTION":
        objs = [o for o in context.selected_objects if o.type == "MESH"]
    elif mode == "COLLECTION":
        col = bpy.data.collections.get(collection_name)
        if col:
            for o in col.all_objects:
                if o.type == "MESH":
                    objs.append(o)
    return objs

def _ensure_box_display(ob):
    ob.display_type = "WIRE"
    ob.show_in_front = True
    ob.hide_render = True
    ob.show_bounds = True
    ob.display_bounds_type = "BOX"

def _ensure_plane_display(ob):
    ob.display_type = "WIRE"
    ob.show_in_front = True
    ob.hide_render = True

def _selection_bbox(objs):
    """World-space AABB of evaluated selection (modifiers applied)."""
    if not objs:
        return Vector((0, 0, 0)), Vector((1, 1, 1))
    deps = bpy.context.evaluated_depsgraph_get()
    mins = Vector(( 1e18,  1e18,  1e18))
    maxs = Vector((-1e18, -1e18, -1e18))
    for ob in objs:
        ob_eval = ob.evaluated_get(deps)
        if ob_eval.type == "MESH":
            me = ob_eval.to_mesh(preserve_all_data_layers=False)
            if me and me.vertices:
                mw = ob_eval.matrix_world
                for v in me.vertices:
                    wv = mw @ v.co
                    mins.x = min(mins.x, wv.x); mins.y = min(mins.y, wv.y); mins.z = min(mins.z, wv.z)
                    maxs.x = max(maxs.x, wv.x); maxs.y = max(maxs.y, wv.y); maxs.z = max(maxs.z, wv.z)
                ob_eval.to_mesh_clear()
                continue
        try:
            bb = ob_eval.bound_box
            mw = ob_eval.matrix_world
            for c in bb:
                wv = mw @ Vector(c)
                mins.x = min(mins.x, wv.x); mins.y = min(mins.y, wv.y); mins.z = min(mins.z, wv.z)
                maxs.x = max(maxs.x, wv.x); maxs.y = max(maxs.y, wv.y); maxs.z = max(maxs.z, wv.z)
        except Exception:
            pass
    size = (maxs - mins)
    size.x = max(size.x, 0.001); size.y = max(size.y, 0.001); size.z = max(size.z, 0.001)
    center = (maxs + mins) * 0.5
    return center, size

def _set_targets_prop_box(box, objs):
    box[PROP_TARGETS] = ",".join([o.name for o in objs])

def _get_targets_from_box(box):
    names = box.get(PROP_TARGETS, "")
    if not names:
        return []
    out = []
    for n in names.split(","):
        o = bpy.data.objects.get(n)
        if o:
            out.append(o)
    return out

def _set_targets_prop_plane(plane, objs):
    plane[PROP_PTARGETS] = ",".join([o.name for o in objs])

def _get_targets_from_plane(plane):
    names = plane.get(PROP_PTARGETS, "")
    if not names:
        return []
    out = []
    for n in names.split(","):
        o = bpy.data.objects.get(n)
        if o:
            out.append(o)
    return out

def _add_boolean(o, box):
    mod_name = MOD_PREFIX + box.name
    mod = o.modifiers.get(mod_name)
    if not mod:
        mod = o.modifiers.new(mod_name, "BOOLEAN")
    mod.operation = "INTERSECT"
    mod.solver = "EXACT"
    mod.object = box
    try:
        while o.modifiers[0] != mod:
            bpy.ops.object.modifier_move_up({"object": o}, modifier=mod.name)
    except Exception:
        pass

def _remove_boolean(o, box):
    mod_name = MOD_PREFIX + box.name
    mod = o.modifiers.get(mod_name)
    if mod:
        o.modifiers.remove(mod)

def _link_exclusive_to_collection(obj, col):
    """Unlink obj from all collections, then link only to col (safe for Blender 4.5)."""
    for c in tuple(obj.users_collection):
        try:
            c.objects.unlink(obj)
        except Exception:
            pass
    if col.objects.get(obj.name) is None:
        col.objects.link(obj)
    scene_root = bpy.context.scene.collection
    if scene_root.objects.get(obj.name):
        try:
            scene_root.objects.unlink(obj)
        except Exception:
            pass

# -------- Results collections (Box) --------

def _ensure_results_collection_box(box):
    name = f"SBX_Results_{box.name}"
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        if box.users_collection:
            box.users_collection[0].children.link(col)
        else:
            bpy.context.scene.collection.children.link(col)
    return col

def _next_iteration_collections_box(box, face_code):
    root = _ensure_results_collection_box(box)
    base = f"{root.name}__Section_"
    idx = 1
    while True:
        main_name = f"{base}{idx:03d}_{face_code}"
        if bpy.data.collections.get(main_name) is None:
            break
        idx += 1
    col_main = bpy.data.collections.new(main_name); root.children.link(col_main)
    col_cut  = bpy.data.collections.new(main_name + "__Cut");        col_main.children.link(col_cut)
    col_mesh = bpy.data.collections.new(main_name + "__Meshes");     col_main.children.link(col_mesh)
    col_proj = bpy.data.collections.new(main_name + "__Projection"); col_main.children.link(col_proj)

    emp = bpy.data.objects.new(f"SBX_FrameAxes_{idx:03d}_{face_code}", None)
    emp.empty_display_type = "PLAIN_AXES"
    emp.matrix_world = box.matrix_world.copy()
    _link_exclusive_to_collection(emp, col_main)

    iter_tag = f"S{idx:03d}_{face_code}"
    box[PROP_LAST_ITER_BOX]  = main_name
    box[PROP_LAST_ITER_TAGB] = iter_tag
    return col_main, col_cut, col_mesh, col_proj, idx, iter_tag

# -------- Results collections (Plane) --------

def _ensure_results_collection_plane(plane):
    name = f"SBP_Results_{plane.name}"
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        if plane.users_collection:
            plane.users_collection[0].children.link(col)
        else:
            bpy.context.scene.collection.children.link(col)
    return col

def _next_iteration_collections_plane(plane, face_tag):
    root = _ensure_results_collection_plane(plane)
    base = f"{root.name}__Section_"
    idx = 1
    while True:
        main_name = f"{base}{idx:03d}_{face_tag}"
        if bpy.data.collections.get(main_name) is None:
            break
        idx += 1
    col_main  = bpy.data.collections.new(main_name); root.children.link(col_main)
    col_plane = bpy.data.collections.new(main_name + "__Plane");      col_main.children.link(col_plane)
    col_cut   = bpy.data.collections.new(main_name + "__Cut");        col_main.children.link(col_cut)
    col_mesh  = bpy.data.collections.new(main_name + "__Meshes");     col_main.children.link(col_mesh)
    col_elev  = bpy.data.collections.new(main_name + "__Elevation");  col_main.children.link(col_elev)

    plane[PROP_LAST_ITER_PLN]  = main_name
    plane[PROP_LAST_ITER_TAGP] = f"S{idx:03d}_{face_tag}"
    return col_main, col_plane, col_cut, col_mesh, col_elev, idx, plane[PROP_LAST_ITER_TAGP]

# ----------------------- Transforms & frames --------------------------------

def _axis_vectors_world(mw):
    xw = (mw.to_3x3() @ Vector((1, 0, 0))).normalized()
    yw = (mw.to_3x3() @ Vector((0, 1, 0))).normalized()
    zw = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
    return xw, yw, zw

def _apply_dims_with_anchors(box, w, d, h, ax_anchor, ay_anchor, az_anchor):
    mw = box.matrix_world
    origin = mw @ Vector((0, 0, 0))
    xw, yw, zw = _axis_vectors_world(mw)

    cur_w = max(box.dimensions.x, 0.001)
    cur_d = max(box.dimensions.y, 0.001)
    cur_h = max(box.dimensions.z, 0.001)

    px_cur = origin + xw * (cur_w * 0.5); nx_cur = origin - xw * (cur_w * 0.5)
    py_cur = origin + yw * (cur_d * 0.5); ny_cur = origin - yw * (cur_d * 0.5)
    pz_cur = origin + zw * (cur_h * 0.5); nz_cur = origin - zw * (cur_h * 0.5)

    w = max(w, 0.001); d = max(d, 0.001); h = max(h, 0.001)
    box.dimensions = (w, d, h)

    new_origin = origin
    if ax_anchor == "NEG":
        nx_new = new_origin - xw * (w * 0.5); new_origin += (nx_cur - nx_new)
    elif ax_anchor == "POS":
        px_new = new_origin + xw * (w * 0.5); new_origin += (px_cur - px_new)

    if ay_anchor == "NEG":
        ny_new = new_origin - yw * (d * 0.5); new_origin += (ny_cur - ny_new)
    elif ay_anchor == "POS":
        py_new = new_origin + yw * (d * 0.5); new_origin += (py_cur - py_new)

    if az_anchor == "NEG":
        nz_new = new_origin - zw * (h * 0.5); new_origin += (nz_cur - nz_new)
    elif az_anchor == "POS":
        pz_new = new_origin + zw * (h * 0.5); new_origin += (pz_cur - pz_new)

    cur_origin = box.matrix_world @ Vector((0, 0, 0))
    box.location += (new_origin - cur_origin)

def _on_dim_update(self, context):
    try:
        if not context or not getattr(self, "live_link", True):
            return
        box = _get_active_box(context)
        if not box:
            return
        _apply_dims_with_anchors(
            box,
            self.width, self.depth, self.height,
            self.anchor_x, self.anchor_y, self.anchor_z
        )
    except Exception:
        pass

# -------- Live move & rotate (box local axes) --------

def _on_move_update(self, context):
    try:
        if not context or not getattr(self, "live_link", True):
            return
        box = _get_active_box(context)
        if not box:
            return
        dx = self.move_x - self.prev_move_x
        dy = self.move_y - self.prev_move_y
        dz = self.move_z - self.prev_move_z
        if abs(dx) + abs(dy) + abs(dz) < 1e-12:
            return
        xw, yw, zw = _axis_vectors_world(box.matrix_world)
        box.location += xw * dx + yw * dy + zw * dz
        self.prev_move_x = self.move_x
        self.prev_move_y = self.move_y
        self.prev_move_z = self.move_z
    except Exception:
        pass

def _rotate_about_origin_local(box, rx_deg, ry_deg, rz_deg):
    if abs(rx_deg) + abs(ry_deg) + abs(rz_deg) < 1e-12:
        return
    rx = rx_deg * 3.141592653589793 / 180.0
    ry = ry_deg * 3.141592653589793 / 180.0
    rz = rz_deg * 3.141592653589793 / 180.0
    mw = box.matrix_world.copy()
    loc = mw.translation.copy()
    Rloc = Euler((rx, ry, rz), "XYZ").to_matrix().to_4x4()
    box.matrix_world = Matrix.Translation(loc) @ (mw.to_3x3().to_4x4() @ Rloc) @ (mw.to_3x3().to_4x4().inverted()) @ Matrix.Translation(-loc) @ mw

def _on_rot_update(self, context):
    try:
        if not context or not getattr(self, "live_link", True):
            return
        box = _get_active_box(context)
        if not box:
            return
        drx = self.rot_x - self.prev_rot_x
        dry = self.rot_y - self.prev_rot_y
        drz = self.rot_z - self.prev_rot_z
        if abs(drx) + abs(dry) + abs(drz) < 1e-12:
            return
        _rotate_about_origin_local(box, drx, dry, drz)
        self.prev_rot_x = self.rot_x
        self.prev_rot_y = self.rot_y
        self.prev_rot_z = self.rot_z
    except Exception:
        pass

# -------- Plane frame & live size with anchors --------

def _ensure_right_handed_uv(u: Vector, v: Vector, n: Vector):
    """Ensure (u, v, n) is right-handed: if u×v doesn't align with n, flip v."""
    if u.cross(v).dot(n) < 0.0:
        v = -v
    return u.normalized(), v.normalized()

def _plane_frame_from_object(plane_obj):
    mw = plane_obj.matrix_world
    u = (mw.to_3x3() @ Vector((1, 0, 0))).normalized()
    v = (mw.to_3x3() @ Vector((0, 1, 0))).normalized()
    n = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
    u, v = _ensure_right_handed_uv(u, v, n)
    center = mw.translation
    hu = max(0.001, plane_obj.dimensions.x * 0.5)
    hv = max(0.001, plane_obj.dimensions.y * 0.5)
    return {"center": center, "normal": n, "u": u, "v": v, "hu": hu, "hv": hv, "dir_in": n}

def _apply_plane_sizes_with_anchors(plane, su, sv, au, av):
    mw = plane.matrix_world
    origin = mw @ Vector((0, 0, 0))
    u = (mw.to_3x3() @ Vector((1, 0, 0))).normalized()
    v = (mw.to_3x3() @ Vector((0, 1, 0))).normalized()

    cur_su = max(plane.dimensions.x, 0.001)
    cur_sv = max(plane.dimensions.y, 0.001)

    pu_cur = origin + u * (cur_su * 0.5); nu_cur = origin - u * (cur_su * 0.5)
    pv_cur = origin + v * (cur_sv * 0.5); nv_cur = origin - v * (cur_sv * 0.5)

    su = max(su, 0.001); sv = max(sv, 0.001)
    plane.dimensions = (su, sv, plane.dimensions.z)

    new_origin = origin
    if au == "NEG":
        nu_new = new_origin - u * (su * 0.5); new_origin += (nu_cur - nu_new)
    elif au == "POS":
        pu_new = new_origin + u * (su * 0.5); new_origin += (pu_cur - pu_new)
    if av == "NEG":
        nv_new = new_origin - v * (sv * 0.5); new_origin += (nv_cur - nv_new)
    elif av == "POS":
        pv_new = new_origin + v * (sv * 0.5); new_origin += (pv_cur - pv_new)

    cur_origin = plane.matrix_world @ Vector((0, 0, 0))
    plane.location += (new_origin - cur_origin)

def _on_plane_size_update(self, context):
    try:
        if not context or not getattr(self, "plane_live_link", True):
            return
        pl = _get_active_plane(context)
        if not pl:
            return
        _apply_plane_sizes_with_anchors(pl, self.plane_size_u, self.plane_size_v, self.plane_anchor_u, self.plane_anchor_v)
    except Exception:
        pass

# ---------------------------- Mesh & projection utils ------------------------

def _project_to_frame(p_world: Vector, frame):
    d = p_world - frame["center"]
    return d.dot(frame["u"]), d.dot(frame["v"])

def _unproject_from_frame(u: float, v: float, frame):
    return frame["center"] + frame["u"] * u + frame["v"] * v

def _project_point_onto_plane(p: Vector, frame):
    n = frame["normal"]
    return p - n * (p - frame["center"]).dot(n)

def _clip_segment_to_rect(u1, v1, u2, v2, hu, hv):
    du, dv = u2 - u1, v2 - v1
    t0, t1 = 0.0, 1.0

    def clip(p, q, t0, t1):
        if p == 0.0:
            return (t0, t1) if q >= 0.0 else (None, None)
        t = q / p
        if p < 0.0:
            if t > t1: return (None, None)
            if t > t0: t0 = t
        else:
            if t < t0: return (None, None)
            if t < t1: t1 = t
        return (t0, t1)

    for p, q in [(-du, u1 + hu), (du, hu - u1), (-dv, v1 + hv), (dv, hv - v1)]:
        t0, t1 = clip(p, q, t0, t1)
        if t0 is None:
            return None

    return (u1 + du * t0, v1 + dv * t0, u1 + du * t1, v1 + dv * t1)

def _intersect_mesh_with_plane(obj, plane_point_world, plane_normal_world):
    """Return polylines (list[list[Vector]]) where the mesh intersects the plane (evaluated)."""
    deps = bpy.context.evaluated_depsgraph_get()
    ob_eval = obj.evaluated_get(deps)
    me = ob_eval.to_mesh(preserve_all_data_layers=False)

    mw = ob_eval.matrix_world
    inv = mw.inverted()
    plane_co_local = inv @ plane_point_world
    plane_no_local = (inv.to_3x3().transposed() @ plane_normal_world).normalized()

    bm = bmesh.new()
    bm.from_mesh(me)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    res = bmesh.ops.bisect_plane(
        bm,
        geom=geom,
        plane_co=plane_co_local,
        plane_no=plane_no_local,
        use_snap_center=False,
        clear_inner=False,
        clear_outer=False,
    )
    cut_edges = [e for e in res.get("geom_cut", []) if isinstance(e, bmesh.types.BMEdge)]

    segments = []
    for e in cut_edges:
        v1 = mw @ e.verts[0].co
        v2 = mw @ e.verts[1].co
        segments.append((v1, v2))

    bm.free()
    ob_eval.to_mesh_clear()
    return _stitch_segments(segments)

def _all_mesh_edges_world(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ob_eval = obj.evaluated_get(deps)
    me = ob_eval.to_mesh(preserve_all_data_layers=False)
    mw = ob_eval.matrix_world
    edges = []
    for e in me.edges:
        v1 = mw @ me.vertices[e.vertices[0]].co
        v2 = mw @ me.vertices[e.vertices[1]].co
        edges.append((v1, v2))
    ob_eval.to_mesh_clear()
    return edges

def _stitch_segments(segments, tol=1e-6):
    chains = []
    for a, b in segments:
        placed = False
        for ch in chains:
            if (ch[-1] - a).length <= tol:
                ch.append(b); placed = True; break
            if (ch[-1] - b).length <= tol:
                ch.append(a); placed = True; break
            if (ch[0] - a).length <= tol:
                ch.insert(0, b); placed = True; break
            if (ch[0] - b).length <= tol:
                ch.insert(0, a); placed = True; break
        if not placed:
            chains.append([a, b])
    return chains

def _make_curve_from_polyline(points, name="SBX_Line"):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    spl = cu.splines.new("POLY")
    spl.points.add(len(points) - 1)
    for i, p in enumerate(points):
        spl.points[i].co = (p.x, p.y, p.z, 1)
    ob = bpy.data.objects.new(name, cu)
    return ob

def _ensure_closed(points, tol=1e-5):
    if not points:
        return points
    if (points[0] - points[-1]).length > tol:
        points = points + [points[0].copy()]
    return points

def _make_ngon_face(points_world, name="SBX_FillFace"):
    if len(points_world) < 3:
        return None
    bm = bmesh.new()
    verts = [bm.verts.new((p.x, p.y, p.z)) for p in points_world]
    bm.verts.index_update()
    try:
        bm.faces.new(verts)
    except ValueError:
        edges = [bm.edges.new((verts[i], verts[(i+1) % len(verts)])) for i in range(len(verts))]
        bmesh.ops.edgenet_fill(bm, edges=edges)
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    return ob

def _clip_edge_to_slab(a: Vector, b: Vector, face, tmin: float, tmax: float):
    d = b - a
    dir_in = face["dir_in"]
    ta = (a - face["center"]).dot(dir_in)
    tb = (b - face["center"]).dot(dir_in)
    if (ta < tmin and tb < tmin) or (ta > tmax and tb > tmax):
        return None
    denom = (tb - ta)
    qa, qb = a.copy(), b.copy()
    if abs(denom) < 1e-12:
        return (a, b) if (tmin <= ta <= tmax) else None
    if ta < tmin:
        lam = (tmin - ta) / denom; qa = a + d * lam; ta = tmin
    elif ta > tmax:
        lam = (tmax - ta) / denom; qa = a + d * lam; ta = tmax
    if tb < tmin:
        lam = (tmin - tb) / denom; qb = b + d * lam; tb = tmin
    elif tb > tmax:
        lam = (tmax - tb) / denom; qb = b + d * lam; tb = tmax
    if (qb - qa).length <= 1e-9:
        return None
    return (qa, qb)

# -------------------- Freestyle-like outlines helpers ------------------------

def _world_normal_from_face(mw: Matrix, n_local: Vector) -> Vector:
    return (mw.to_3x3().inverted().transposed() @ n_local).normalized()

def _build_world_bvhs(objects):
    """Build BVH trees in world space for simple hidden-line tests."""
    deps = bpy.context.evaluated_depsgraph_get()
    trees = []
    for ob in objects:
        ob_eval = ob.evaluated_get(deps)
        me = ob_eval.to_mesh(preserve_all_data_layers=False)
        if not me:
            continue
        mw = ob_eval.matrix_world
        verts = [mw @ v.co for v in me.vertices]
        if not verts or not me.polygons:
            ob_eval.to_mesh_clear()
            continue
        verts_tuples = [tuple(v) for v in verts]
        polys = [tuple(p.vertices) for p in me.polygons]
        try:
            tree = BVHTree.FromPolygons(verts_tuples, polys)
            trees.append(tree)
        except Exception:
            pass
        ob_eval.to_mesh_clear()
    return trees

def _candidate_outline_edges_world(obj, view_dir_world: Vector):
    """
    Return world-space segments that are either boundary edges (front-facing)
    or silhouette edges (adjacent faces flip front/back) relative to view_dir_world.
    """
    deps = bpy.context.evaluated_depsgraph_get()
    ob_eval = obj.evaluated_get(deps)
    me = ob_eval.to_mesh(preserve_all_data_layers=False)
    if not me:
        return []

    mw = ob_eval.matrix_world
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    segs = []
    cam_dir = view_dir_world.normalized()

    def is_front(n_world: Vector) -> bool:
        return (n_world.dot(cam_dir) < 0.0)

    for e in bm.edges:
        lf = e.link_faces
        if len(lf) == 0:
            continue  # non-manifold
        elif len(lf) == 1:
            f = lf[0]
            nW = _world_normal_from_face(mw, f.normal)
            if is_front(nW):
                v1 = mw @ e.verts[0].co
                v2 = mw @ e.verts[1].co
                segs.append((v1, v2))
        elif len(lf) == 2:
            f1, f2 = lf
            n1 = _world_normal_from_face(mw, f1.normal)
            n2 = _world_normal_from_face(mw, f2.normal)
            if is_front(n1) != is_front(n2):
                v1 = mw @ e.verts[0].co
                v2 = mw @ e.verts[1].co
                segs.append((v1, v2))
    bm.free()
    ob_eval.to_mesh_clear()
    return segs

def _estimate_max_depth_along_dir(objects, frame, dir_sign: float):
    """If plane_depth <= 0, estimate a safe far depth from bounds."""
    dir_vec = frame["normal"] * dir_sign
    maxd = 0.0
    for ob in objects:
        mw = ob.matrix_world
        try:
            for c in ob.bound_box:
                w = mw @ Vector(c)
                d = (w - frame["center"]).dot(dir_vec)
                if d > maxd:
                    maxd = d
        except Exception:
            pass
    return max(maxd, frame["hu"] + frame["hv"])

# ---- HLR (split segments by visibility) ----

def _ray_occludes_point(p: Vector, frame, to_plane: Vector, bvh_trees, maxdist: float, eps: float) -> bool:
    if not bvh_trees or maxdist <= eps:
        return False
    origin = p + to_plane * eps
    dist = max(maxdist - eps * 0.5, eps)
    for tree in bvh_trees:
        hit = tree.ray_cast(origin, to_plane, dist)
        if hit and hit[0] is not None:
            return True
    return False

def _split_segment_by_visibility(wp1: Vector, wp2: Vector, frame, camera_dir: Vector,
                                 bvh_trees, samples: int = 13,
                                 eps_scale: float = 1e-4, min_len_scale: float = 1e-5):
    to_plane = (-camera_dir).normalized()
    span = (frame["hu"] + frame["hv"])
    seg_len = (wp2 - wp1).length
    if seg_len <= 1e-9:
        return []

    base = max(5, samples)
    extra = int(max(0, (seg_len / max(span, 1e-6)) * 64))
    samples_eff = min(101, max(base, extra))

    eps = max(1e-6, eps_scale * span)
    min_len = max(1e-6, min_len_scale * span)

    def dist_to_plane(p: Vector) -> float:
        return (frame["center"] - p).dot(to_plane)

    ts = [i / (samples_eff - 1) for i in range(samples_eff)]
    pts = [wp1.lerp(wp2, t) for t in ts]

    vis = []
    for p in pts:
        d = dist_to_plane(p)
        if d <= eps * 2.0:
            vis.append(True)
            continue
        occluded = _ray_occludes_point(p, frame, to_plane, bvh_trees, d, eps)
        vis.append(not occluded)

    out = []
    i = 0
    while i < len(ts) - 1:
        if vis[i]:
            j = i + 1
            while j < len(ts) and vis[j]:
                j += 1
            a_t = ts[i]
            b_t = ts[j - 1]

            if i > 0 and not vis[i - 1]:
                lo, hi = ts[i - 1], ts[i]
                for _ in range(6):
                    mid = (lo + hi) * 0.5
                    pm = wp1.lerp(wp2, mid)
                    d = dist_to_plane(pm)
                    occ = (d > eps * 2.0) and _ray_occludes_point(pm, frame, to_plane, bvh_trees, d, eps)
                    if occ: lo = mid
                    else:   hi = mid
                a_t = hi

            if j < len(ts) and not vis[j]:
                lo, hi = ts[j - 1], ts[j]
                for _ in range(6):
                    mid = (lo + hi) * 0.5
                    pm = wp1.lerp(wp2, mid)
                    d = dist_to_plane(pm)
                    occ = (d > eps * 2.0) and _ray_occludes_point(pm, frame, to_plane, bvh_trees, d, eps)
                    if occ: hi = mid
                    else:   lo = mid
                b_t = lo

            pa = wp1.lerp(wp2, a_t)
            pb = wp1.lerp(wp2, b_t)
            if (pb - pa).length >= min_len:
                out.append((pa, pb))
            i = j
        else:
            i += 1
    return out

# ------------- Helpers to map clipped UV back to 3D along a segment ---------

def _t_from_uv(u1, v1, u2, v2, uc, vc):
    du = u2 - u1; dv = v2 - v1
    if abs(du) >= abs(dv) and abs(du) > 1e-12:
        return (uc - u1) / du
    elif abs(dv) > 1e-12:
        return (vc - v1) / dv
    return 0.0

# ---- outline generation (3D-first HLR, then project) -----------------------

def _outline_segments_for_plane(targets, frame, dir_sign: float):
    camera_dir = frame["normal"] * dir_sign
    eps = max(1e-6, 1e-5 * (frame["hu"] + frame["hv"]))
    tmin = eps

    s = bpy.context.scene.sbx_settings
    base_depth = max(0.0, getattr(s, "plane_depth", 0.0))
    if base_depth <= 0.0:
        tmax = max(eps, _estimate_max_depth_along_dir(targets, frame, dir_sign))
    else:
        tmax = max(eps, base_depth)

    trees = _build_world_bvhs(targets)
    out = []
    tested = 0

    for t in targets:
        cand = _candidate_outline_edges_world(t, camera_dir)
        for (a, b) in cand:
            clipped = _clip_edge_to_plane_depth(a, b, frame, tmin, tmax, dir_sign=dir_sign)
            if not clipped:
                continue
            qa, qb = clipped

            pa = _project_point_onto_plane(qa, frame)
            pb = _project_point_onto_plane(qb, frame)
            u1, v1 = _project_to_frame(pa, frame)
            u2, v2 = _project_to_frame(pb, frame)
            c = _clip_segment_to_rect(u1, v1, u2, v2, frame["hu"], frame["hv"])
            if not c:
                continue
            cu1, cv1, cu2, cv2 = c

            tA = max(0.0, min(1.0, _t_from_uv(u1, v1, u2, v2, cu1, cv1)))
            tB = max(0.0, min(1.0, _t_from_uv(u1, v1, u2, v2, cu2, cv2)))
            qA = qa.lerp(qb, tA)
            qB = qa.lerp(qb, tB)

            tested += 1
            visible_3d = _split_segment_by_visibility(
                qA, qB, frame, camera_dir, trees,
                samples=getattr(s, "hlr_samples", 13),
                eps_scale=getattr(s, "hlr_eps", 1e-4),
                min_len_scale=1e-5
            )
            for p3, q3 in visible_3d:
                p2 = _project_point_onto_plane(p3, frame)
                q2 = _project_point_onto_plane(q3, frame)
                out.append((p2, q2))

    if not out:
        if tested == 0:
            elev_segments = []
            if tmax > tmin + 1e-9:
                for t in targets:
                    edges = _all_mesh_edges_world(t)
                    for (a, b) in edges:
                        clipped = _clip_edge_to_plane_depth(a, b, frame, tmin, tmax, dir_sign=dir_sign)
                        if not clipped:
                            continue
                        qa, qb = clipped
                        pa = _project_point_onto_plane(qa, frame)
                        pb = _project_point_onto_plane(qb, frame)
                        u1, v1 = _project_to_frame(pa, frame)
                        u2, v2 = _project_to_frame(pb, frame)
                        c = _clip_segment_to_rect(u1, v1, u2, v2, frame["hu"], frame["hv"])
                        if not c:
                            continue
                        cu1, cv1, cu2, cv2 = c
                        p2 = _unproject_from_frame(cu1, cv1, frame)
                        q2 = _unproject_from_frame(cu2, cv2, frame)
                        elev_segments.append((p2, q2))
            return _stitch_segments(elev_segments, tol=1e-6)
        else:
            return []

    return _stitch_segments(out, tol=1e-6)

def _outline_segments_for_box(targets, face):
    camera_dir = face["dir_in"]
    eps = max(1e-6, 1e-5 * (face["hu"] + face["hv"]))
    tmin = eps
    tmax = max(eps, face["thickness"] - eps)

    trees = _build_world_bvhs(targets)
    out = []
    tested = 0

    for t in targets:
        cand = _candidate_outline_edges_world(t, camera_dir)
        for (a, b) in cand:
            clipped = _clip_edge_to_slab(a, b, face, tmin, tmax)
            if not clipped:
                continue
            qa, qb = clipped

            pa = _project_point_onto_plane(qa, face)
            pb = _project_point_onto_plane(qb, face)
            u1, v1 = _project_to_frame(pa, face)
            u2, v2 = _project_to_frame(pb, face)
            c = _clip_segment_to_rect(u1, v1, u2, v2, face["hu"], face["hv"])
            if not c:
                continue
            cu1, cv1, cu2, cv2 = c

            tA = max(0.0, min(1.0, _t_from_uv(u1, v1, u2, v2, cu1, cv1)))
            tB = max(0.0, min(1.0, _t_from_uv(u1, v1, u2, v2, cu2, cv2)))
            qA = qa.lerp(qb, tA)
            qB = qa.lerp(qb, tB)

            tested += 1
            s = bpy.context.scene.sbx_settings
            visible_3d = _split_segment_by_visibility(
                qA, qB, face, camera_dir, trees,
                samples=getattr(s, "hlr_samples", 13),
                eps_scale=getattr(s, "hlr_eps", 1e-4),
                min_len_scale=1e-5
            )
            for p3, q3 in visible_3d:
                p2 = _project_point_onto_plane(p3, face)
                q2 = _project_point_onto_plane(q3, face)
                out.append((p2, q2))

    if not out:
        if tested == 0:
            proj_segments = []
            for t in targets:
                for (a, b) in _all_mesh_edges_world(t):
                    clipped = _clip_edge_to_slab(a, b, face, tmin, tmax)
                    if not clipped:
                        continue
                    qa, qb = clipped
                    pa = _project_point_onto_plane(qa, face)
                    pb = _project_point_onto_plane(qb, face)
                    u1, v1 = _project_to_frame(pa, face)
                    u2, v2 = _project_to_frame(pb, face)
                    c = _clip_segment_to_rect(u1, v1, u2, v2, face["hu"], face["hv"])
                    if not c:
                        continue
                    cu1, cv1, cu2, cv2 = c
                    p2 = _unproject_from_frame(cu1, cv1, face)
                    q2 = _unproject_from_frame(cu2, cv2, face)
                    proj_segments.append((p2, q2))
            return _stitch_segments(proj_segments, tol=1e-6)
        else:
            return []

    return _stitch_segments(out, tol=1e-6)

# --------- Frame objects for 2D export (view-aligned) -----------------------

def _create_face_frame_empty(face, iter_tag, col_main, n_view=None):
    """Create an Empty whose local XY matches the section plane in VIEW orientation."""
    u = face["u"]; v = face["v"]; n = n_view if n_view is not None else face["normal"]
    u, v = _ensure_right_handed_uv(u, v, n)
    c = face["center"]
    mw = Matrix((
        (u.x, v.x, n.x, c.x),
        (u.y, v.y, n.y, c.y),
        (u.z, v.z, n.z, c.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    emp = bpy.data.objects.new(f"SBX_Frame2D_{iter_tag}", None)
    emp.empty_display_type = "ARROWS"
    emp.matrix_world = mw
    _link_exclusive_to_collection(emp, col_main)
    return emp

def _create_plane_frame_empty(frame, iter_tag, col_main, n_view):
    """Plane 2D frame, view-aligned (for planar sections)."""
    u = frame["u"]; v = frame["v"]; n = n_view
    u, v = _ensure_right_handed_uv(u, v, n)
    c = frame["center"]
    mw = Matrix((
        (u.x, v.x, n.x, c.x),
        (u.y, v.y, n.y, c.y),
        (u.z, v.z, n.z, c.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    emp = bpy.data.objects.new(f"SBP_Frame2D_{iter_tag}", None)
    emp.empty_display_type = "ARROWS"
    emp.matrix_world = mw
    _link_exclusive_to_collection(emp, col_main)
    return emp

# ------------------------------- Properties ---------------------------------

class SBX_Settings(PropertyGroup):
    # Box targeting
    target_mode: EnumProperty(
        name="Targets",
        items=[("SELECTION", "Selection", "Use current selection"),
               ("COLLECTION", "Collection", "Use objects in a collection")],
        default="SELECTION")
    collection_name: StringProperty(name="Collection", description="Target collection name")
    live_link: BoolProperty(name="Live Link (Box)", default=True, description="Live update for box size/move/rotate")

    # Box: anchors & dims (live)
    anchor_x: EnumProperty(name="X Anchor", items=[("CENTER","Center","Grow both"),("NEG","-X","Grow +X"),("POS","+X","Grow -X")], default="CENTER")
    anchor_y: EnumProperty(name="Y Anchor", items=[("CENTER","Center","Grow both"),("NEG","-Y","Grow +Y"),("POS","+Y","Grow -Y")], default="CENTER")
    anchor_z: EnumProperty(name="Z Anchor", items=[("CENTER","Center","Grow both"),("NEG","-Z","Grow +Z (bottom fixed)"),("POS","+Z","Grow -Z (top fixed)")], default="NEG")

    width:  FloatProperty(name="Width",  default=1.0, min=0.001, update=_on_dim_update)
    depth:  FloatProperty(name="Depth",  default=1.0, min=0.001, update=_on_dim_update)
    height: FloatProperty(name="Height", default=1.0, min=0.001, update=_on_dim_update)

    # Box: face for sectioning
    face_choice: EnumProperty(name="Face", items=[("PX","+X",""),("NX","-X",""),("PY","+Y",""),("NY","-Y",""),("PZ","+Z",""),("NZ","-Z","")], default="PZ")

    # Box: live move
    move_x: FloatProperty(name="Move X", default=0.0, update=_on_move_update)
    move_y: FloatProperty(name="Move Y", default=0.0, update=_on_move_update)
    move_z: FloatProperty(name="Move Z", default=0.0, update=_on_move_update)
    prev_move_x: FloatProperty(name="prev_move_x", default=0.0)
    prev_move_y: FloatProperty(name="prev_move_y", default=0.0)
    prev_move_z: FloatProperty(name="prev_move_z", default=0.0)

    # Box: live rotate (degrees, local axes)
    rot_x: FloatProperty(name="Rot X°", default=0.0, update=_on_rot_update)
    rot_y: FloatProperty(name="Rot Y°", default=0.0, update=_on_rot_update)
    rot_z: FloatProperty(name="Rot Z°", default=0.0, update=_on_rot_update)
    prev_rot_x: FloatProperty(name="prev_rot_x", default=0.0)
    prev_rot_y: FloatProperty(name="prev_rot_y", default=0.0)
    prev_rot_z: FloatProperty(name="prev_rot_z", default=0.0)

    # Planar: live link & anchors
    plane_live_link: BoolProperty(name="Plane Live Link", default=True)
    plane_anchor_u: EnumProperty(name="U Anchor", items=[("CENTER","Center","Grow both"),("NEG","-U","Grow +U"),("POS","+U","Grow -U")], default="CENTER")
    plane_anchor_v: EnumProperty(name="V Anchor", items=[("CENTER","Center","Grow both"),("NEG","-V","Grow +V"),("POS","+V","Grow -V")], default="CENTER")

    plane_orient: EnumProperty(name="Plane", items=[("XY","XY",""),("XZ","XZ",""),("YZ","YZ","")], default="XY")
    plane_size_u: FloatProperty(name="Size U", default=5.0, min=0.01, update=_on_plane_size_update)
    plane_size_v: FloatProperty(name="Size V", default=5.0, min=0.01, update=_on_plane_size_update)
    plane_offset: FloatProperty(name="Offset", default=0.0, description="Move active plane along its normal by this amount")

    plane_dir: EnumProperty(name="Direction", items=[("POS","+Normal","toward +N"),("NEG","-Normal","toward -N")], default="POS")
    plane_depth: FloatProperty(name="Depth", default=2.0, min=0.0, description="Elevation slab thickness from plane along chosen direction")

    # HLR tuning (shared by box & planar)
    hlr_samples: IntProperty(
        name="Visibility Samples",
        description="Samples along each edge for visibility splitting",
        default=13, min=5, max=101
    )
    hlr_eps: FloatProperty(
        name="HLR Epsilon",
        description="Ray offset scale relative to frame span (avoid self-hits)",
        default=1e-4, min=1e-7, max=1e-2, precision=6
    )

    # Export paths (per type)
    export_path_box: StringProperty(name="Export Path (Box)", subtype="FILE_PATH", default="")
    export_path_plane: StringProperty(name="Export Path (Plane)", subtype="FILE_PATH", default="")

    # Export axis flips (affects SVG & DXF)
    export_flip_x: BoolProperty(name="Flip X (export)", default=False)
    export_flip_y: BoolProperty(name="Flip Y (export)", default=False)

# -------------------------------- BOX Operators ------------------------------

class SBX_OT_create_from_selection(Operator):
    bl_idname = "sbx.create_from_selection"
    bl_label = "Create Section Box from Selection"
    bl_options = {"REGISTER", "UNDO"}
    padding: FloatProperty(name="Padding", default=0.01, min=0.0)

    def execute(self, context):
        sel = list(context.selected_objects)
        if not sel:
            self.report({"WARNING"}, "Select at least one object")
            return {"CANCELLED"}
        center, size = _selection_bbox(sel)
        size += Vector((self.padding, self.padding, self.padding)) * 2.0
        bpy.ops.mesh.primitive_cube_add(location=center)
        box = context.active_object
        box.name = BOX_PREFIX
        _ensure_box_display(box)
        box.scale = size * 0.5

        s = context.scene.sbx_settings
        s.width, s.depth, s.height = box.dimensions.x, box.dimensions.y, box.dimensions.z

        s.prev_move_x = s.move_x
        s.prev_move_y = s.move_y
        s.prev_move_z = s.move_z
        s.prev_rot_x  = s.rot_x
        s.prev_rot_y  = s.rot_y
        s.prev_rot_z  = s.rot_z

        mesh_targets = [o for o in sel if o.type == "MESH"]
        if mesh_targets:
            _set_targets_prop_box(box, mesh_targets)
            for o in mesh_targets:
                _add_boolean(o, box)
        else:
            _set_targets_prop_box(box, [])
            self.report({"INFO"}, "No mesh targets in selection; box created. Bind meshes later if needed.")
        return {"FINISHED"}

class SBX_OT_bind_targets(Operator):
    bl_idname = "sbx.bind_targets"
    bl_label = "Bind Targets to Active Box"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(items=[("SELECTION","Selection","Bind selected objects"),
                              ("COLLECTION","Collection","Bind a collection")], default="SELECTION")
    collection_name: StringProperty(name="Collection")

    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({"WARNING"}, "Select a SectionBox first")
            return {"CANCELLED"}
        objs = _objects_from_target_mode(context, self.mode, self.collection_name)
        if not objs:
            self.report({"WARNING"}, "No valid objects to bind")
            return {"CANCELLED"}
        _set_targets_prop_box(box, objs)
        for o in objs:
            _add_boolean(o, box)
        return {"FINISHED"}

class SBX_OT_toggle_live(Operator):
    bl_idname = "sbx.toggle_live"
    bl_label = "Toggle Live Booleans"
    bl_options = {"REGISTER", "UNDO"}
    enable: BoolProperty(default=True)

    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({"WARNING"}, "Select a SectionBox first")
            return {"CANCELLED"}
        for o in _get_targets_from_box(box):
            mod = o.modifiers.get(MOD_PREFIX + box.name)
            if mod:
                mod.show_viewport = self.enable
                mod.show_render = self.enable
        return {"FINISHED"}

class SBX_OT_apply(Operator):
    bl_idname = "sbx.apply"
    bl_label = "Bake/Apply Booleans"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({"WARNING"}, "Select a SectionBox first")
            return {"CANCELLED"}
        for o in _get_targets_from_box(box):
            mod_name = MOD_PREFIX + box.name
            mod = o.modifiers.get(mod_name)
            if mod:
                try:
                    bpy.context.view_layer.objects.active = o
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                except Exception as e:
                    self.report({"WARNING"}, f"Failed to apply on {o.name}: {e}")
        return {"FINISHED"}

class SBX_OT_cleanup(Operator):
    bl_idname = "sbx.cleanup"
    bl_label = "Remove Box + Mods"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({"WARNING"}, "Select a SectionBox first")
            return {"CANCELLED"}
        for o in _get_targets_from_box(box):
            _remove_boolean(o, box)
        bpy.data.objects.remove(box, do_unlink=True)
        return {"FINISHED"}

# -------- Box face frames --------

def _box_face_frames(box: bpy.types.Object):
    mw = box.matrix_world
    xw, yw, zw = _axis_vectors_world(mw)
    origin = mw @ Vector((0, 0, 0))
    dims = box.dimensions
    frames = {
        "PX": {"center": origin + xw * (dims.x * 0.5), "normal":  xw, "u": yw, "v": zw, "hu": dims.y*0.5, "hv": dims.z*0.5, "thickness": dims.x, "dir_in": -xw},
        "NX": {"center": origin - xw * (dims.x * 0.5), "normal": -xw, "u": yw, "v": zw, "hu": dims.y*0.5, "hv": dims.z*0.5, "thickness": dims.x, "dir_in":  xw},
        "PY": {"center": origin + yw * (dims.y * 0.5), "normal":  yw, "u": xw, "v": zw, "hu": dims.x*0.5, "hv": dims.z*0.5, "thickness": dims.y, "dir_in": -yw},
        "NY": {"center": origin - yw * (dims.y * 0.5), "normal": -yw, "u": xw, "v": zw, "hu": dims.x*0.5, "hv": dims.z*0.5, "thickness": dims.y, "dir_in":  yw},
        "PZ": {"center": origin + zw * (dims.z * 0.5), "normal":  zw, "u": xw, "v": yw, "hu": dims.x*0.5, "hv": dims.y*0.5, "thickness": dims.z, "dir_in": -zw},
        "NZ": {"center": origin - zw * (dims.z * 0.5), "normal": -zw, "u": xw, "v": yw, "hu": dims.x*0.5, "hv": dims.y*0.5, "thickness": dims.z, "dir_in":  zw},
    }
    # enforce right-handed UV for each face against its own normal
    for k, f in frames.items():
        u, v = _ensure_right_handed_uv(f["u"], f["v"], f["normal"])
        f["u"], f["v"] = u, v
    return frames

class SBX_OT_section_lines(Operator):
    bl_idname = "sbx.section_lines"
    bl_label = "Generate Box Section (selected face)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({"WARNING"}, "Select a SectionBox first")
            return {"CANCELLED"}
        targets = _get_targets_from_box(box)
        if not targets:
            self.report({"WARNING"}, "No targets bound to this SectionBox")
            return {"CANCELLED"}
        s = context.scene.sbx_settings
        frames = _box_face_frames(box)
        face = frames.get(s.face_choice)
        if not face:
            self.report({"WARNING"}, "Invalid face selection")
            return {"CANCELLED"}

        col_main, col_cut, col_mesh, col_proj, _idx, iter_tag = _next_iteration_collections_box(box, s.face_choice)

        # store a view-aligned frame Empty for 2D export
        _create_face_frame_empty(face, iter_tag, col_main, n_view=face["dir_in"])

        diag = box.dimensions.length
        eps = max(1e-6, diag * 1e-5)

        made_cut = 0
        made_proj = 0
        made_faces = 0

        for t in targets:
            plane_point = face["center"] + face["dir_in"] * eps
            cut_polylines = _intersect_mesh_with_plane(t, plane_point, face["normal"])

            cut_segments = []
            for line in cut_polylines:
                for i in range(len(line) - 1):
                    p1, p2 = line[i], line[i + 1]
                    u1, v1 = _project_to_frame(p1, face)
                    u2, v2 = _project_to_frame(p2, face)
                    c = _clip_segment_to_rect(u1, v1, u2, v2, face["hu"], face["hv"])
                    if not c:
                        continue
                    cu1, cv1, cu2, cv2 = c
                    wp1 = _unproject_from_frame(cu1, cv1, face)
                    wp2 = _unproject_from_frame(cu2, cv2, face)
                    cut_segments.append((wp1, wp2))

            stitched_cut = _stitch_segments(cut_segments, tol=1e-6)
            for pts in stitched_cut:
                if len(pts) < 2:
                    continue
                cu = _make_curve_from_polyline(pts, name=f"SBX_Line_cut_{iter_tag}_{t.name}")
                _link_exclusive_to_collection(cu, col_cut)
                made_cut += 1

                closed = _ensure_closed(pts, tol=1e-5)
                if len(closed) >= 4 and (closed[0] - closed[-1]).length <= 1e-4:
                    face_obj = _make_ngon_face(closed, name=f"SBX_FillFace_{iter_tag}_{t.name}")
                    if face_obj:
                        _link_exclusive_to_collection(face_obj, col_mesh)
                        made_faces += 1

        stitched_proj = _outline_segments_for_box(targets, face)
        for pts in stitched_proj:
            if len(pts) < 2:
                continue
            cu = _make_curve_from_polyline(pts, name=f"SBX_Line_projection_{iter_tag}")
            _link_exclusive_to_collection(cu, col_proj)
            made_proj += 1

        self.report({"INFO"}, f"Box {iter_tag}: {made_cut} cut, {made_proj} proj, {made_faces} faces")
        return {"FINISHED"}

# ------------------------------- PLANE logic --------------------------------

def _ensure_planes_root():
    name = "SBP_Planes"
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

class SBP_OT_create_plane(Operator):
    bl_idname = "sbp.create_plane"
    bl_label = "Create Cutting Plane"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.sbx_settings
        bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
        pl = context.active_object
        pl.name = PLANE_PREFIX
        _ensure_plane_display(pl)

        if s.plane_orient == "XY":
            pl.rotation_euler = (0.0, 0.0, 0.0)            # normal +Z
        elif s.plane_orient == "XZ":
            pl.rotation_euler = (0.0, 1.57079632679, 0.0)  # normal +Y
        else:  # YZ
            pl.rotation_euler = (1.57079632679, 0.0, 0.0)  # normal +X

        _apply_plane_sizes_with_anchors(pl, s.plane_size_u, s.plane_size_v, s.plane_anchor_u, s.plane_anchor_v)
        root = _ensure_planes_root()
        _link_exclusive_to_collection(pl, root)
        s.plane_size_u, s.plane_size_v = pl.dimensions.x, pl.dimensions.y
        _set_targets_prop_plane(pl, [])
        self.report({"INFO"}, f"Created plane: {s.plane_orient}")
        return {"FINISHED"}

class SBP_OT_offset_plane(Operator):
    bl_idname = "sbp.offset_plane"
    bl_label = "Offset Active Plane"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        pl = _get_active_plane(context)
        if not pl:
            self.report({"WARNING"}, "Select a SectionPlane first")
            return {"CANCELLED"}
        s = context.scene.sbx_settings
        frame = _plane_frame_from_object(pl)
        pl.location += frame["normal"] * s.plane_offset
        s.plane_offset = 0.0
        return {"FINISHED"}

class SBP_OT_bind_targets(Operator):
    bl_idname = "sbp.bind_targets"
    bl_label = "Bind Selection to Active Plane"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        pl = _get_active_plane(context)
        if not pl:
            self.report({"WARNING"}, "Select a SectionPlane first")
            return {"CANCELLED"}
        meshes = [o for o in context.selected_objects if o.type == "MESH"]
        if not meshes:
            self.report({"WARNING"}, "Select mesh objects to bind")
            return {"CANCELLED"}
        _set_targets_prop_plane(pl, meshes)
        self.report({"INFO"}, f"Bound {len(meshes)} mesh(es) to plane")
        return {"FINISHED"}

def _clip_edge_to_plane_depth(a: Vector, b: Vector, frame, tmin: float, tmax: float, dir_sign: float):
    d = b - a
    dir_vec = frame["normal"] * dir_sign
    ta = (a - frame["center"]).dot(dir_vec)
    tb = (b - frame["center"]).dot(dir_vec)
    if (ta < tmin and tb < tmin) or (ta > tmax and tb > tmax):
        return None
    denom = (tb - ta)
    qa, qb = a.copy(), b.copy()
    if abs(denom) < 1e-12:
        return (a, b) if (tmin <= ta <= tmax) else None
    if ta < tmin:
        lam = (tmin - ta) / denom; qa = a + d * lam; ta = tmin
    elif ta > tmax:
        lam = (tmax - ta) / denom; qa = a + d * lam; ta = tmax
    if tb < tmin:
        lam = (tmin - tb) / denom; qb = b + d * lam; tb = tmin
    elif tb > tmax:
        lam = (tmax - tb) / denom; qb = b + d * lam; tb = tmax
    if (qb - qa).length <= 1e-9:
        return None
    return (qa, qb)

class SBP_OT_generate(Operator):
    bl_idname = "sbp.generate"
    bl_label = "Generate Planar Section"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        pl = _get_active_plane(context)
        if not pl:
            self.report({"WARNING"}, "Select a SectionPlane first")
            return {"CANCELLED"}
        targets = _get_targets_from_plane(pl)
        if not targets:
            targets = [o for o in context.selected_objects if o.type == "MESH"]
            if not targets:
                self.report({"WARNING"}, "No targets bound/selected for planar section")
                return {"CANCELLED"}

        s = context.scene.sbx_settings
        frame = _plane_frame_from_object(pl)
        face_tag = f"{s.plane_orient}{'P' if s.plane_dir == 'POS' else 'N'}"
        col_main, col_plane, col_cut, col_mesh, col_elev, _idx, iter_tag = _next_iteration_collections_plane(pl, face_tag)

        # Freeze a copy of the plane into the iteration
        pl_copy_me = pl.data.copy()
        pl_copy = bpy.data.objects.new(f"SBP_Plane_{iter_tag}", pl_copy_me)
        pl_copy.matrix_world = pl.matrix_world.copy()
        _link_exclusive_to_collection(pl_copy, col_plane)

        dir_sign = 1.0 if s.plane_dir == "POS" else -1.0
        n_view = frame["normal"] * dir_sign
        # view-aligned frame for export
        _create_plane_frame_empty(frame, iter_tag, col_main, n_view=n_view)

        eps = max(1e-6, 1e-5 * (s.plane_size_u + s.plane_size_v))

        made_cut = 0
        made_elev = 0
        made_faces = 0

        for t in targets:
            plane_point = frame["center"] + frame["normal"] * (eps * dir_sign * 0.5)
            cut_polylines = _intersect_mesh_with_plane(t, plane_point, frame["normal"])

            cut_segments = []
            for line in cut_polylines:
                for i in range(len(line) - 1):
                    p1, p2 = line[i], line[i + 1]
                    u1, v1 = _project_to_frame(p1, frame)
                    u2, v2 = _project_to_frame(p2, frame)
                    c = _clip_segment_to_rect(u1, v1, u2, v2, frame["hu"], frame["hv"])
                    if not c:
                        continue
                    cu1, cv1, cu2, cv2 = c
                    wp1 = _unproject_from_frame(cu1, cv1, frame)
                    wp2 = _unproject_from_frame(cu2, cv2, frame)
                    cut_segments.append((wp1, wp2))

            stitched_cut = _stitch_segments(cut_segments, tol=1e-6)
            for pts in stitched_cut:
                if len(pts) < 2:
                    continue
                cu = _make_curve_from_polyline(pts, name=f"SBP_Line_cut_{iter_tag}_{t.name}")
                _link_exclusive_to_collection(cu, col_cut)
                made_cut += 1

                closed = _ensure_closed(pts, tol=1e-5)
                if len(closed) >= 4 and (closed[0] - closed[-1]).length <= 1e-4:
                    face_obj = _make_ngon_face(closed, name=f"SBP_FillFace_{iter_tag}_{t.name}")
                    if face_obj:
                        _link_exclusive_to_collection(face_obj, col_mesh)
                        made_faces += 1

        stitched_elev = _outline_segments_for_plane(targets, frame, dir_sign)
        for pts in stitched_elev:
            if len(pts) < 2:
                continue
            cu = _make_curve_from_polyline(pts, name=f"SBP_Line_elev_{iter_tag}")
            _link_exclusive_to_collection(cu, col_elev)
            made_elev += 1

        self.report({"INFO"}, f"Plane {iter_tag}: {made_cut} cut, {made_elev} elev, {made_faces} faces")
        return {"FINISHED"}

# ------------------------------ SVG / DXF Export -----------------------------

def _normalize_export_path(path, ext, default_name):
    if not path:
        return bpy.path.abspath(f"//{default_name}.{ext}")
    p = bpy.path.abspath(path)
    if p.endswith(("/", "\\")) or os.path.isdir(p):
        p = os.path.join(p, f"{default_name}.{ext}")
    dot_ext = f".{ext}".lower()
    if not p.lower().endswith(dot_ext):
        p = bpy.path.ensure_ext(p, dot_ext)
    dirpath = os.path.dirname(p) or bpy.path.abspath("//")
    os.makedirs(dirpath, exist_ok=True)
    return p

# --- helpers to locate a frame object and project to its XY ---

def _find_frame_matrix_in_collection(root_collection):
    """Search the iteration tree for a 2D frame object and return its world matrix."""
    frame_mw = None
    def walk(col):
        nonlocal frame_mw
        for ob in col.objects:
            nm = ob.name
            if nm.startswith("SBX_Frame2D_") or nm.startswith("SBP_Frame2D_"):
                frame_mw = ob.matrix_world.copy(); return
            if nm.startswith("SBP_Plane_"):
                # fallback if no dedicated frame found
                if frame_mw is None:
                    frame_mw = ob.matrix_world.copy()
        for ch in col.children:
            if frame_mw is None:
                walk(ch)
    walk(root_collection)
    return frame_mw

def _project_on_frame_xy(points3d, world_to_frame):
    out = []
    for p in points3d:
        lp = world_to_frame @ p
        out.append((lp.x, lp.y))
    return out

def _collect_from_collection_tree(root_collection, kinds=("cut","projection","elevation","fills")):
    polylines = {"cut": [], "projection": [], "elevation": []}
    fill_polys = []
    def walk(col):
        for ob in col.objects:
            nm = ob.name
            if ob.type == "CURVE":
                which = None
                if nm.startswith("SBX_Line_cut_") or nm.startswith("SBP_Line_cut_"):
                    which = "cut"
                elif nm.startswith("SBX_Line_projection_"):
                    which = "projection"
                elif nm.startswith("SBP_Line_elev_"):
                    which = "elevation"
                if which and which in polylines:
                    mw = ob.matrix_world.copy()
                    for sp in ob.data.splines:
                        if sp.type != "POLY" or len(sp.points) < 2:
                            continue
                        pts = [mw @ Vector((p.co.x, p.co.y, p.co.z)) for p in sp.points]
                        polylines[which].append(pts)
            elif ob.type == "MESH":
                if nm.startswith("SBX_FillFace_") or nm.startswith("SBP_FillFace_"):
                    mw = ob.matrix_world; me = ob.data
                    me.calc_loop_triangles()
                    for poly in me.polygons:
                        verts = [mw @ me.vertices[i].co for i in poly.vertices]
                        if len(verts) >= 3:
                            fill_polys.append(verts)
        for ch in col.children:
            walk(ch)
    walk(root_collection)
    return polylines, fill_polys

def _export_svg_from_collection(root_collection, filepath):
    s = bpy.context.scene.sbx_settings
    flip_x = bool(getattr(s, "export_flip_x", False))
    flip_y = bool(getattr(s, "export_flip_y", False))

    lines, fills = _collect_from_collection_tree(root_collection)
    cut_lines  = lines["cut"]
    proj_lines = lines["projection"]
    elev_lines = lines["elevation"]

    frame_mw = _find_frame_matrix_in_collection(root_collection)
    world_to_frame = frame_mw.inverted() if frame_mw else None

    all2d = []
    def add2d_from3d(pl):
        nonlocal all2d
        if world_to_frame:
            all2d += _project_on_frame_xy(pl, world_to_frame)
        else:
            all2d += [(p.x, p.y) for p in pl]

    for pl in cut_lines + proj_lines + elev_lines:
        add2d_from3d(pl)
    for poly in fills:
        add2d_from3d(poly)

    if not all2d:
        return False, "Nothing to export in this iteration"

    xs = [x for (x, _) in all2d]; ys = [y for (_, y) in all2d]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    w = max(1.0, maxx - minx); h = max(1.0, maxy - miny)

    svg = ET.Element("svg", attrib={"xmlns":"http://www.w3.org/2000/svg","width":str(w),"height":str(h),"viewBox":f"0 0 {w} {h}"})

    def map_xy(x, y):
        sx = (maxx - x) if flip_x else (x - minx)
        sy = (maxy - y) if flip_y else (y - miny)
        return sx, sy

    def project_points(pl):
        if world_to_frame:
            pts2 = _project_on_frame_xy(pl, world_to_frame)
        else:
            pts2 = [(p.x, p.y) for p in pl]
        out = []
        for (x, y) in pts2:
            sx, sy = map_xy(x, y)
            out.append(f"{sx},{sy}")
        return " ".join(out)

    if fills:
        g_fills = ET.SubElement(svg, "g", attrib={"id":"fills"})
        for poly in fills:
            ET.SubElement(g_fills, "polygon", attrib={"points": project_points(poly), "fill":"#b3b3b3", "stroke":"none"})

    stroke_w = "0.25"

    if cut_lines:
        g_cut = ET.SubElement(svg, "g", attrib={"id":"cut"})
        for pl in cut_lines:
            ET.SubElement(g_cut, "polyline", attrib={"points": project_points(pl), "fill":"none", "stroke":"black", "stroke-width":stroke_w})

    if proj_lines:
        g_proj = ET.SubElement(svg, "g", attrib={"id":"projection"})
        for pl in proj_lines:
            ET.SubElement(g_proj, "polyline", attrib={"points": project_points(pl), "fill":"none", "stroke":"black", "stroke-width":stroke_w})

    if elev_lines:
        g_elev = ET.SubElement(svg, "g", attrib={"id":"elevation"})
        for pl in elev_lines:
            ET.SubElement(g_elev, "polyline", attrib={"points": project_points(pl), "fill":"none", "stroke":"black", "stroke-width":stroke_w})

    ET.ElementTree(svg).write(filepath, encoding="utf-8", xml_declaration=False)
    return True, f"SVG saved: {filepath}"

def _export_dxf_from_collection(root_collection, filepath):
    s = bpy.context.scene.sbx_settings
    flip_x = bool(getattr(s, "export_flip_x", False))
    flip_y = bool(getattr(s, "export_flip_y", False))

    lines, _fills = _collect_from_collection_tree(root_collection)
    polylines3d = lines["cut"] + lines["projection"] + lines["elevation"]
    if not polylines3d:
        return False, "No lines to export in this iteration"

    frame_mw = _find_frame_matrix_in_collection(root_collection)
    world_to_frame = frame_mw.inverted() if frame_mw else Matrix.Identity(4)

    # Project all to 2D first (global extents)
    polylines2d = []
    all2d = []
    for pl in polylines3d:
        pts2 = _project_on_frame_xy(pl, world_to_frame)
        polylines2d.append(pts2)
        all2d.extend(pts2)

    xs = [p[0] for p in all2d]; ys = [p[1] for p in all2d]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)

    def map_xy(x, y):
        sx = (maxx - x) if flip_x else (x - minx)
        sy = (maxy - y) if flip_y else (y - miny)
        return sx, sy

    def w(f, s): f.write(s); f.write("\n")
    with open(filepath, "w", encoding="utf-8") as f:
        w(f,"0"); w(f,"SECTION"); w(f,"2"); w(f,"ENTITIES")
        for pl in polylines2d:
            for i in range(len(pl) - 1):
                x1, y1 = map_xy(pl[i][0], pl[i][1])
                x2, y2 = map_xy(pl[i+1][0], pl[i+1][1])
                w(f,"0"); w(f,"LINE")
                w(f,"8"); w(f,"SECTION")
                w(f,"10"); w(f, str(float(x1)))
                w(f,"20"); w(f, str(float(y1)))
                w(f,"30"); w(f,"0.0")
                w(f,"11"); w(f, str(float(x2)))
                w(f,"21"); w(f, str(float(y2)))
                w(f,"31"); w(f,"0.0")
        w(f,"0"); w(f,"ENDSEC"); w(f,"0"); w(f,"EOF")
    return True, f"DXF saved: {filepath}"

# -- Export operators (per-iteration, per-type) --

class SBX_OT_export_box_svg(Operator):
    bl_idname = "sbx.export_box_svg"
    bl_label = "Export Box SVG (last run)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        s = context.scene.sbx_settings
        box = _get_active_box(context)
        if not box:
            self.report({"WARNING"}, "Select a SectionBox that has generated results")
            return {"CANCELLED"}
        iter_name = box.get(PROP_LAST_ITER_BOX, "")
        iter_tag  = box.get(PROP_LAST_ITER_TAGB, "S001")
        if not iter_name:
            self.report({"WARNING"}, "No box iteration found. Generate a section first.")
            return {"CANCELLED"}
        col = bpy.data.collections.get(iter_name)
        if not col:
            self.report({"WARNING"}, "Stored box iteration collection not found")
            return {"CANCELLED"}
        default_name = f"sbx_{iter_tag}"
        path = _normalize_export_path(s.export_path_box, "svg", default_name)
        ok, msg = _export_svg_from_collection(col, path)
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED"}

class SBX_OT_export_box_dxf(Operator):
    bl_idname = "sbx.export_box_dxf"
    bl_label = "Export Box DXF (last run)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        s = context.scene.sbx_settings
        box = _get_active_box(context)
        if not box:
            self.report({"WARNING"}, "Select a SectionBox that has generated results")
            return {"CANCELLED"}
        iter_name = box.get(PROP_LAST_ITER_BOX, "")
        iter_tag  = box.get(PROP_LAST_ITER_TAGB, "S001")
        if not iter_name:
            self.report({"WARNING"}, "No box iteration found. Generate a section first.")
            return {"CANCELLED"}
        col = bpy.data.collections.get(iter_name)
        if not col:
            self.report({"WARNING"}, "Stored box iteration collection not found")
            return {"CANCELLED"}
        default_name = f"sbx_{iter_tag}"
        path = _normalize_export_path(s.export_path_box, "dxf", default_name)
        ok, msg = _export_dxf_from_collection(col, path)
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED"}

class SBP_OT_export_plane_svg(Operator):
    bl_idname = "sbp.export_plane_svg"
    bl_label = "Export Plane SVG (last run)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        s = context.scene.sbx_settings
        pl = _get_active_plane(context)
        if not pl:
            self.report({"WARNING"}, "Select a SectionPlane that has generated results")
            return {"CANCELLED"}
        iter_name = pl.get(PROP_LAST_ITER_PLN, "")
        iter_tag  = pl.get(PROP_LAST_ITER_TAGP, "S001")
        if not iter_name:
            self.report({"WARNING"}, "No plane iteration found. Generate a planar section first.")
            return {"CANCELLED"}
        col = bpy.data.collections.get(iter_name)
        if not col:
            self.report({"WARNING"}, "Stored plane iteration collection not found")
            return {"CANCELLED"}
        default_name = f"sbp_{iter_tag}"
        path = _normalize_export_path(s.export_path_plane, "svg", default_name)
        ok, msg = _export_svg_from_collection(col, path)
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED"}

class SBP_OT_export_plane_dxf(Operator):
    bl_idname = "sbp.export_plane_dxf"
    bl_label = "Export Plane DXF (last run)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        s = context.scene.sbx_settings
        pl = _get_active_plane(context)
        if not pl:
            self.report({"WARNING"}, "Select a SectionPlane that has generated results")
            return {"CANCELLED"}
        iter_name = pl.get(PROP_LAST_ITER_PLN, "")
        iter_tag  = pl.get(PROP_LAST_ITER_TAGP, "S001")
        if not iter_name:
            self.report({"WARNING"}, "No plane iteration found. Generate a planar section first.")
            return {"CANCELLED"}
        col = bpy.data.collections.get(iter_name)
        if not col:
            self.report({"WARNING"}, "Stored plane iteration collection not found")
            return {"CANCELLED"}
        default_name = f"sbp_{iter_tag}"
        path = _normalize_export_path(s.export_path_plane, "dxf", default_name)
        ok, msg = _export_dxf_from_collection(col, path)
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED"}

# ----------------------------------- UI -------------------------------------

class SBX_PT_panel(Panel):
    bl_label = "Section Toolbox"
    bl_idname = "SBX_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Section Toolbox"

    def draw(self, context):
        s = context.scene.sbx_settings
        layout = self.layout

        # Create box
        col = layout.column(align=True)
        col.label(text="Create Box")
        col.operator("sbx.create_from_selection")

        # Live box params
        col.separator()
        col.label(text="Section Box Dimensions")
        col.prop(s, "live_link")
        row = col.row(align=True); row.prop(s, "width"); row.prop(s, "depth"); row.prop(s, "height")
        row = col.row(align=True); row.prop(s, "anchor_x"); row.prop(s, "anchor_y"); row.prop(s, "anchor_z")

        # Live Move / Rotate
        col.separator()
        col.label(text="Move")
        row = col.row(align=True)
        row.prop(s, "move_x"); row.prop(s, "move_y"); row.prop(s, "move_z")
        col.label(text="Rotate (live, local axes)")
        row = col.row(align=True)
        row.prop(s, "rot_x"); row.prop(s, "rot_y"); row.prop(s, "rot_z")

        # Bind / Live boolean
        col.separator()
        col.label(text="Bind / Targets (Box)")
        row = col.row(align=True)
        op = row.operator("sbx.bind_targets", text="Bind Selection"); op.mode = "SELECTION"
        op = row.operator("sbx.bind_targets", text="Bind Collection"); op.mode = "COLLECTION"
        col.prop(s, "collection_name", text="Collection")

        col.separator()
        col.label(text="Live Boolean Control")
        row = col.row(align=True)
        op = row.operator("sbx.toggle_live", text="Enable Live");  op.enable = True
        op = row.operator("sbx.toggle_live", text="Disable Live"); op.enable = False

        # Box Section
        col.separator()
        col.label(text="Box Sections")
        col.prop(s, "face_choice")
        col.operator("sbx.section_lines", text="Generate Box Section")

        # HLR tuning (shared)
        col.separator()
        col.label(text="Hidden Line Removal")
        col.prop(s, "hlr_samples")
        col.prop(s, "hlr_eps")

        # ---------------- Planar subset ----------------
        layout.separator()
        box_ui = layout.box()
        box_ui.label(text="Planar Sections")
        box_ui.prop(s, "plane_live_link", text="Plane Live Link")
        row = box_ui.row(align=True)
        row.prop(s, "plane_orient"); row.prop(s, "plane_dir"); row.prop(s, "plane_depth")
        row2 = box_ui.row(align=True)
        row2.prop(s, "plane_size_u"); row2.prop(s, "plane_size_v")
        row3 = box_ui.row(align=True)
        row3.prop(s, "plane_anchor_u"); row3.prop(s, "plane_anchor_v")
        row4 = box_ui.row(align=True)
        row4.operator("sbp.create_plane", text="Create Plane")
        row4.prop(s, "plane_offset")
        row4.operator("sbp.offset_plane", text="Apply Offset")
        row5 = box_ui.row(align=True)
        row5.operator("sbp.bind_targets", text="Bind Selection to Plane")
        row6 = box_ui.row(align=True)
        row6.operator("sbp.generate", text="Generate Planar Section")

        # Export (per type + last iteration)
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Export Box (last iteration)")
        col.prop(s, "export_path_box")
        row = col.row(align=True); row.operator("sbx.export_box_svg"); row.operator("sbx.export_box_dxf")

        col.separator()
        col.label(text="Export Plane (last iteration)")
        col.prop(s, "export_path_plane")
        row = col.row(align=True); row.operator("sbp.export_plane_svg"); row.operator("sbp.export_plane_dxf")

        # Export options
        layout.separator()
        exp = layout.box()
        exp.label(text="Export Options (SVG/DXF)")
        exp.prop(s, "export_flip_x")
        exp.prop(s, "export_flip_y")

        # Finalise
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Bake/Remove")
        col.operator("sbx.apply", text="Bake / Apply")
        col.operator("sbx.cleanup", text="Remove Box + Mods", icon="TRASH")

# ------------------------------ Registration --------------------------------

classes = (
    SBX_Settings,

    SBX_OT_create_from_selection,
    SBX_OT_bind_targets,
    SBX_OT_toggle_live,
    SBX_OT_apply,
    SBX_OT_cleanup,
    SBX_OT_section_lines,

    SBP_OT_create_plane,
    SBP_OT_offset_plane,
    SBP_OT_bind_targets,
    SBP_OT_generate,

    SBX_OT_export_box_svg,
    SBX_OT_export_box_dxf,
    SBP_OT_export_plane_svg,
    SBP_OT_export_plane_dxf,

    SBX_PT_panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.sbx_settings = PointerProperty(type=SBX_Settings)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    if hasattr(bpy.types.Scene, "sbx_settings"):
        del bpy.types.Scene.sbx_settings

if __name__ == "__main__":
    register()

