# Section Box + Planar Sections (live transforms, versioned results, SVG/DXF 2D export)

bl_info = {
    "name": "Section Toolbox",
    "author": "Victor Calixto",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
    "location": "3D Viewport > N-panel > Section Box",
    "description": "Parametric box & planar sections with per-face cuts/projections, filled faces, versioned collections, SVG/DXF 2D export",
    "category": "3D View",
}

import bpy
import bmesh
import os
from mathutils import Vector, Matrix
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    BoolProperty,
    StringProperty,
    EnumProperty,
    PointerProperty,
    FloatProperty,
)
import xml.etree.ElementTree as ET

# ------------------------------- Constants ----------------------------------

BOX_PREFIX    = 'SectionBox'
PLANE_PREFIX  = 'SectionPlane'
MOD_PREFIX    = 'SBX_'
PROP_TARGETS  = 'sbx_targets'
PROP_PTARGETS = 'sbp_targets'

# -------------------------------- Utilities ---------------------------------

def _get_active_box(context):
    ob = context.active_object
    if ob and ob.type == 'MESH' and ob.name.startswith(BOX_PREFIX):
        return ob
    return None

def _get_active_plane(context):
    ob = context.active_object
    if ob and ob.type == 'MESH' and ob.name.startswith(PLANE_PREFIX):
        return ob
    return None

def _objects_from_target_mode(context, mode='SELECTION', collection_name=''):
    objs = []
    if mode == 'SELECTION':
        objs = [o for o in context.selected_objects if o.type == 'MESH']
    elif mode == 'COLLECTION':
        col = bpy.data.collections.get(collection_name)
        if col:
            for o in col.all_objects:
                if o.type == 'MESH':
                    objs.append(o)
    return objs

def _ensure_box_display(ob):
    ob.display_type = 'WIRE'
    ob.show_in_front = True
    ob.hide_render = True
    ob.show_bounds = True
    ob.display_bounds_type = 'BOX'

def _ensure_plane_display(ob):
    ob.display_type = 'WIRE'
    ob.show_in_front = True
    ob.hide_render = True

def _selection_bbox(objs):
    """World-space AABB of evaluated selection (modifiers applied)."""
    if not objs:
        return Vector((0,0,0)), Vector((1,1,1))
    deps = bpy.context.evaluated_depsgraph_get()
    mins = Vector(( 1e18,  1e18,  1e18))
    maxs = Vector((-1e18, -1e18, -1e18))
    for ob in objs:
        ob_eval = ob.evaluated_get(deps)
        if ob_eval.type == 'MESH':
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
    box[PROP_TARGETS] = ','.join([o.name for o in objs])

def _get_targets_from_box(box):
    names = box.get(PROP_TARGETS, '')
    if not names: return []
    out = []
    for n in names.split(','):
        o = bpy.data.objects.get(n)
        if o: out.append(o)
    return out

def _set_targets_prop_plane(plane, objs):
    plane[PROP_PTARGETS] = ','.join([o.name for o in objs])

def _get_targets_from_plane(plane):
    names = plane.get(PROP_PTARGETS, '')
    if not names: return []
    out = []
    for n in names.split(','):
        o = bpy.data.objects.get(n)
        if o: out.append(o)
    return out

def _add_boolean(o, box):
    mod_name = MOD_PREFIX + box.name
    mod = o.modifiers.get(mod_name)
    if not mod:
        mod = o.modifiers.new(mod_name, 'BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.solver = 'EXACT'
    mod.object = box
    # try to keep it near top
    try:
        while o.modifiers[0] != mod:
            bpy.ops.object.modifier_move_up({'object': o}, modifier=mod.name)
    except Exception:
        pass

def _remove_boolean(o, box):
    mod_name = MOD_PREFIX + box.name
    mod = o.modifiers.get(mod_name)
    if mod:
        o.modifiers.remove(mod)

def _link_exclusive_to_collection(obj, col):
    """Link only to 'col' (unlink everywhere else including scene root)."""
    for c in list(obj.users_collection):
        try:
            c.objects.unlink(obj)
        except Exception:
            pass
    try:
        col.objects.link(obj)
    except Exception:
        # already linked
        pass
    # Ensure not in master scene root
    try:
        if obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(obj)
    except Exception:
        pass

def _walk_collection_objects(col, out_list):
    for ob in col.objects:
        out_list.append(ob)
    for child in col.children:
        _walk_collection_objects(child, out_list)

# -------- Results Collections (Box) --------

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
    iter_tag = f"S{idx:03d}_{face_code}"
    return col_main, col_cut, col_mesh, col_proj, idx, iter_tag

def _find_latest_iteration_box(box):
    root = _ensure_results_collection_box(box)
    latest = None
    latest_idx = -1
    for child in root.children:
        n = child.name
        if "__Section_" in n:
            try:
                after = n.split("__Section_")[1]
                num = int(after[:3])
                if num > latest_idx:
                    latest_idx = num
                    latest = child
            except Exception:
                pass
    return latest, latest_idx

# -------- Results Collections (Plane) --------

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
    col_main = bpy.data.collections.new(main_name); root.children.link(col_main)
    col_plane = bpy.data.collections.new(main_name + "__Plane");      col_main.children.link(col_plane)
    col_cut   = bpy.data.collections.new(main_name + "__Cut");        col_main.children.link(col_cut)
    col_mesh  = bpy.data.collections.new(main_name + "__Meshes");     col_main.children.link(col_mesh)
    col_elev  = bpy.data.collections.new(main_name + "__Elevation");  col_main.children.link(col_elev)
    iter_tag = f"S{idx:03d}_{face_tag}"
    return col_main, col_plane, col_cut, col_mesh, col_elev, idx, iter_tag

def _find_latest_iteration_plane(plane):
    root = _ensure_results_collection_plane(plane)
    latest = None
    latest_idx = -1
    for child in root.children:
        n = child.name
        if "__Section_" in n:
            try:
                after = n.split("__Section_")[1]
                num = int(after[:3])
                if num > latest_idx:
                    latest_idx = num
                    latest = child
            except Exception:
                pass
    return latest, latest_idx

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
    if ax_anchor == 'NEG':
        nx_new = new_origin - xw * (w * 0.5); new_origin += (nx_cur - nx_new)
    elif ax_anchor == 'POS':
        px_new = new_origin + xw * (w * 0.5); new_origin += (px_cur - px_new)
    if ay_anchor == 'NEG':
        ny_new = new_origin - yw * (d * 0.5); new_origin += (ny_cur - ny_new)
    elif ay_anchor == 'POS':
        py_new = new_origin + yw * (d * 0.5); new_origin += (py_cur - py_new)
    if az_anchor == 'NEG':
        nz_new = new_origin - zw * (h * 0.5); new_origin += (nz_cur - nz_new)
    elif az_anchor == 'POS':
        pz_new = new_origin + zw * (h * 0.5); new_origin += (pz_cur - pz_new)

    cur_origin = box.matrix_world @ Vector((0, 0, 0))
    box.location += (new_origin - cur_origin)

def _on_dim_update(self, context):
    try:
        if not context or not getattr(self, 'live_link', True): return
        box = _get_active_box(context)
        if not box: return
        _apply_dims_with_anchors(box, self.width, self.depth, self.height, self.anchor_x, self.anchor_y, self.anchor_z)
    except Exception:
        pass

# --- live move & rotate ---

def _rotate_about_origin_local(obj, axis_world: Vector, degrees: float):
    if abs(degrees) < 1e-9: return
    rad = degrees * 3.141592653589793 / 180.0
    mw = obj.matrix_world.copy()
    loc = mw.translation.copy()
    mw.translation = Vector((0,0,0))
    R = Matrix.Rotation(rad, 4, axis_world.normalized())
    obj.matrix_world = Matrix.Translation(loc) @ R @ Matrix.Translation(-loc) @ mw

def _on_move_update(self, context):
    try:
        if not context: return
        box = _get_active_box(context)
        if not box: return
        dx = self.move_x - self.prev_move_x
        dy = self.move_y - self.prev_move_y
        dz = self.move_z - self.prev_move_z
        if dx or dy or dz:
            box.location += Vector((dx, dy, dz))
            self.prev_move_x = self.move_x
            self.prev_move_y = self.move_y
            self.prev_move_z = self.move_z
    except Exception:
        pass

def _on_rot_update(self, context):
    try:
        if not context: return
        box = _get_active_box(context)
        if not box: return
        mw = box.matrix_world
        xw, yw, zw = _axis_vectors_world(mw)
        rx = self.rot_x - self.prev_rot_x
        ry = self.rot_y - self.prev_rot_y
        rz = self.rot_z - self.prev_rot_z
        if rx: _rotate_about_origin_local(box, xw, rx)
        if ry: _rotate_about_origin_local(box, yw, ry)
        if rz: _rotate_about_origin_local(box, zw, rz)
        self.prev_rot_x = self.rot_x
        self.prev_rot_y = self.rot_y
        self.prev_rot_z = self.rot_z
    except Exception:
        pass

# -------- Plane frame & live size with anchors --------

def _plane_frame_from_object(plane_obj):
    mw = plane_obj.matrix_world
    u = (mw.to_3x3() @ Vector((1,0,0))).normalized()
    v = (mw.to_3x3() @ Vector((0,1,0))).normalized()
    n = (mw.to_3x3() @ Vector((0,0,1))).normalized()
    center = mw.translation
    hu = max(0.001, plane_obj.dimensions.x * 0.5)
    hv = max(0.001, plane_obj.dimensions.y * 0.5)
    return {'center': center, 'normal': n, 'u': u, 'v': v, 'hu': hu, 'hv': hv, 'dir_in': n}

def _apply_plane_sizes_with_anchors(plane, su, sv, au, av):
    mw = plane.matrix_world
    origin = mw @ Vector((0, 0, 0))
    u = (mw.to_3x3() @ Vector((1,0,0))).normalized()
    v = (mw.to_3x3() @ Vector((0,1,0))).normalized()

    cur_su = max(plane.dimensions.x, 0.001)
    cur_sv = max(plane.dimensions.y, 0.001)

    pu_cur = origin + u * (cur_su * 0.5); nu_cur = origin - u * (cur_su * 0.5)
    pv_cur = origin + v * (cur_sv * 0.5); nv_cur = origin - v * (cur_sv * 0.5)

    su = max(su, 0.001); sv = max(sv, 0.001)
    plane.dimensions = (su, sv, plane.dimensions.z)

    new_origin = origin
    if au == 'NEG':
        nu_new = new_origin - u * (su * 0.5); new_origin += (nu_cur - nu_new)
    elif au == 'POS':
        pu_new = new_origin + u * (su * 0.5); new_origin += (pu_cur - pu_new)
    if av == 'NEG':
        nv_new = new_origin - v * (sv * 0.5); new_origin += (nv_cur - nv_new)
    elif av == 'POS':
        pv_new = new_origin + v * (sv * 0.5); new_origin += (pv_cur - pv_new)

    cur_origin = plane.matrix_world @ Vector((0,0,0))
    plane.location += (new_origin - cur_origin)

def _on_plane_size_update(self, context):
    try:
        if not context or not getattr(self, 'plane_live_link', True): return
        pl = _get_active_plane(context)
        if not pl: return
        _apply_plane_sizes_with_anchors(pl, self.plane_size_u, self.plane_size_v, self.plane_anchor_u, self.plane_anchor_v)
    except Exception:
        pass

# ---------------------------- Mesh & projection utils ------------------------

def _project_to_frame(p_world: Vector, frame):
    d = p_world - frame['center']
    return d.dot(frame['u']), d.dot(frame['v'])

def _unproject_from_frame(u: float, v: float, frame):
    return frame['center'] + frame['u'] * u + frame['v'] * v

def _project_point_onto_plane(p: Vector, frame):
    n = frame['normal']
    return p - n * (p - frame['center']).dot(n)

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

def _intersect_mesh_with_plane(obj: bpy.types.Object, plane_point_world: Vector, plane_normal_world: Vector):
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
    cut_edges = [e for e in res.get('geom_cut', []) if isinstance(e, bmesh.types.BMEdge)]

    segments = []
    for e in cut_edges:
        v1 = mw @ e.verts[0].co
        v2 = mw @ e.verts[1].co
        segments.append((v1, v2))

    bm.free()
    ob_eval.to_mesh_clear()
    return _stitch_segments(segments)

def _all_mesh_edges_world(obj: bpy.types.Object):
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

def _make_curve_from_polyline(points, name='SBX_Line'):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    spl = cu.splines.new('POLY')
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
    dir_in = face['dir_in']
    ta = (a - face['center']).dot(dir_in)
    tb = (b - face['center']).dot(dir_in)
    if (ta < tmin and tb < tmin) or (ta > tmax and tb > tmax): return None
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
    if (qb - qa).length <= 1e-9: return None
    return (qa, qb)

# ------------------------------- Properties ---------------------------------

class SBX_Settings(PropertyGroup):
    # Box targeting
    target_mode: EnumProperty(
        name='Targets',
        items=[('SELECTION', 'Selection', 'Use current selection'),
               ('COLLECTION', 'Collection', 'Use objects in a collection')],
        default='SELECTION')
    collection_name: StringProperty(name='Collection', description='Target collection name')
    live_enabled: BoolProperty(name='Live', default=True)

    # Box: live & anchors
    live_link: BoolProperty(name='Live link to active box', default=True)
    anchor_x: EnumProperty(name='X Anchor', items=[('CENTER','Center','Grow both'),('NEG','-X','Grow +X'),('POS','+X','Grow -X')], default='CENTER')
    anchor_y: EnumProperty(name='Y Anchor', items=[('CENTER','Center','Grow both'),('NEG','-Y','Grow +Y'),('POS','+Y','Grow -Y')], default='CENTER')
    anchor_z: EnumProperty(name='Z Anchor', items=[('CENTER','Center','Grow both'),('NEG','-Z','Grow +Z (bottom fixed)'),('POS','+Z','Grow -Z (top fixed)')], default='NEG')

    # Box: dims
    width:  FloatProperty(name='Width',  default=1.0, min=0.001, update=_on_dim_update)
    depth:  FloatProperty(name='Depth',  default=1.0, min=0.001, update=_on_dim_update)
    height: FloatProperty(name='Height', default=1.0, min=0.001, update=_on_dim_update)

    # Box: face
    face_choice: EnumProperty(name='Face', items=[('PX','+X',''),('NX','-X',''),('PY','+Y',''),('NY','-Y',''),('PZ','+Z',''),('NZ','-Z','')], default='PZ')

    # Box: live move/rotate (no apply button)
    move_x: FloatProperty(name='Move X', default=0.0, update=_on_move_update)
    move_y: FloatProperty(name='Move Y', default=0.0, update=_on_move_update)
    move_z: FloatProperty(name='Move Z', default=0.0, update=_on_move_update)
    prev_move_x: FloatProperty(name='prev_move_x', default=0.0)
    prev_move_y: FloatProperty(name='prev_move_y', default=0.0)
    prev_move_z: FloatProperty(name='prev_move_z', default=0.0)

    rot_x: FloatProperty(name='Rot X°', default=0.0, update=_on_rot_update)
    rot_y: FloatProperty(name='Rot Y°', default=0.0, update=_on_rot_update)
    rot_z: FloatProperty(name='Rot Z°', default=0.0, update=_on_rot_update)
    prev_rot_x: FloatProperty(name='prev_rot_x', default=0.0)
    prev_rot_y: FloatProperty(name='prev_rot_y', default=0.0)
    prev_rot_z: FloatProperty(name='prev_rot_z', default=0.0)

    # Planar: live link & anchors for plane
    plane_live_link: BoolProperty(name='Plane Live Link', default=True, description='Changing Size U/V updates the active SectionPlane immediately')
    plane_anchor_u: EnumProperty(name='U Anchor', items=[('CENTER','Center','Grow both'),('NEG','-U','Grow +U'),('POS','+U','Grow -U')], default='CENTER')
    plane_anchor_v: EnumProperty(name='V Anchor', items=[('CENTER','Center','Grow both'),('NEG','-V','Grow +V'),('POS','+V','Grow -V')], default='CENTER')

    # Planar: creation & generation settings
    plane_orient: EnumProperty(name='Plane', items=[('XY','XY',''),('XZ','XZ',''),('YZ','YZ','')], default='XY')
    plane_size_u: FloatProperty(name='Size U', default=5.0, min=0.01, update=_on_plane_size_update)
    plane_size_v: FloatProperty(name='Size V', default=5.0, min=0.01, update=_on_plane_size_update)
    plane_offset: FloatProperty(name='Offset', default=0.0, description='Move active plane along its normal by this amount')

    plane_dir: EnumProperty(name='Direction', items=[('POS','+Normal','toward +N'),('NEG','-Normal','toward -N')], default='POS')
    plane_depth: FloatProperty(name='Depth', default=2.0, min=0.0, description='Elevation slab thickness from plane along chosen direction')

    # Export (2D only, per last iteration of active box/plane)
    export_path: StringProperty(name='Base File Path', subtype='FILE_PATH', default='')

# -------------------------------- BOX Operators ------------------------------

class SBX_OT_create_from_selection(Operator):
    bl_idname = 'sbx.create_from_selection'
    bl_label = 'Create Section Box from Selection'
    bl_options = {'REGISTER', 'UNDO'}
    padding: FloatProperty(name='Padding', default=0.01, min=0.0)
    def execute(self, context):
        sel = list(context.selected_objects)
        if not sel:
            self.report({'WARNING'}, 'Select at least one object')
            return {'CANCELLED'}
        center, size = _selection_bbox(sel)
        size += Vector((self.padding, self.padding, self.padding)) * 2.0
        bpy.ops.mesh.primitive_cube_add(location=center)
        box = context.active_object
        box.name = BOX_PREFIX
        _ensure_box_display(box)
        # default cube has size 2 -> set scale so dimensions == desired size
        box.scale = size * 0.5
        s = context.scene.sbx_settings
        s.width, s.depth, s.height = box.dimensions.x, box.dimensions.y, box.dimensions.z
        # Init live move/rot baselines (so first change is a delta from 0)
        s.prev_move_x = s.prev_move_y = s.prev_move_z = 0.0
        s.prev_rot_x = s.prev_rot_y = s.prev_rot_z = 0.0
        # Bind selected meshes automatically
        mesh_targets = [o for o in sel if o.type == 'MESH']
        if mesh_targets:
            _set_targets_prop_box(box, mesh_targets)
            for o in mesh_targets: _add_boolean(o, box)
        else:
            _set_targets_prop_box(box, [])
            self.report({'INFO'}, 'No mesh targets in selection; box created. Bind meshes later if needed.')
        return {'FINISHED'}

class SBX_OT_bind_targets(Operator):
    bl_idname = 'sbx.bind_targets'
    bl_label = 'Bind Targets to Active Box'
    bl_options = {'REGISTER', 'UNDO'}
    mode: EnumProperty(items=[('SELECTION','Selection','Bind selected objects'),
                              ('COLLECTION','Collection','Bind a collection')], default='SELECTION')
    collection_name: StringProperty(name='Collection')
    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({'WARNING'}, 'Select a SectionBox first')
            return {'CANCELLED'}
        objs = _objects_from_target_mode(context, self.mode, self.collection_name)
        if not objs:
            self.report({'WARNING'}, 'No valid objects to bind')
            return {'CANCELLED'}
        _set_targets_prop_box(box, objs)
        for o in objs: _add_boolean(o, box)
        return {'FINISHED'}

class SBX_OT_toggle_live(Operator):
    bl_idname = 'sbx.toggle_live'
    bl_label = 'Toggle Live Booleans'
    bl_options = {'REGISTER', 'UNDO'}
    enable: BoolProperty(default=True)
    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({'WARNING'}, 'Select a SectionBox first')
            return {'CANCELLED'}
        for o in _get_targets_from_box(box):
            mod = o.modifiers.get(MOD_PREFIX + box.name)
            if mod:
                mod.show_viewport = self.enable
                mod.show_render = self.enable
        return {'FINISHED'}

class SBX_OT_apply(Operator):
    bl_idname = 'sbx.apply'
    bl_label = 'Bake/Apply Booleans'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({'WARNING'}, 'Select a SectionBox first')
            return {'CANCELLED'}
        for o in _get_targets_from_box(box):
            mod_name = MOD_PREFIX + box.name
            mod = o.modifiers.get(mod_name)
            if mod:
                try:
                    bpy.context.view_layer.objects.active = o
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                except Exception as e:
                    self.report({'WARNING'}, f'Failed to apply on {o.name}: {e}')
        return {'FINISHED'}

class SBX_OT_cleanup(Operator):
    bl_idname = 'sbx.cleanup'
    bl_label = 'Remove Box + Mods'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({'WARNING'}, 'Select a SectionBox first')
            return {'CANCELLED'}
        for o in _get_targets_from_box(box):
            _remove_boolean(o, box)
        bpy.data.objects.remove(box, do_unlink=True)
        return {'FINISHED'}

# --- Box face frames ---

def _box_face_frames(box: bpy.types.Object):
    mw = box.matrix_world
    xw, yw, zw = _axis_vectors_world(mw)
    origin = mw @ Vector((0, 0, 0))
    dims = box.dimensions
    frames = {
        'PX': {'center': origin + xw * (dims.x * 0.5), 'normal':  xw, 'u': yw, 'v': zw, 'hu': dims.y*0.5, 'hv': dims.z*0.5, 'thickness': dims.x, 'dir_in': -xw},
        'NX': {'center': origin - xw * (dims.x * 0.5), 'normal': -xw, 'u': yw, 'v': zw, 'hu': dims.y*0.5, 'hv': dims.z*0.5, 'thickness': dims.x, 'dir_in':  xw},
        'PY': {'center': origin + yw * (dims.y * 0.5), 'normal':  yw, 'u': xw, 'v': zw, 'hu': dims.x*0.5, 'hv': dims.z*0.5, 'thickness': dims.y, 'dir_in': -yw},
        'NY': {'center': origin - yw * (dims.y * 0.5), 'normal': -yw, 'u': xw, 'v': zw, 'hu': dims.x*0.5, 'hv': dims.z*0.5, 'thickness': dims.y, 'dir_in':  yw},
        'PZ': {'center': origin + zw * (dims.z * 0.5), 'normal':  zw, 'u': xw, 'v': yw, 'hu': dims.x*0.5, 'hv': dims.y*0.5, 'thickness': dims.z, 'dir_in': -zw},
        'NZ': {'center': origin - zw * (dims.z * 0.5), 'normal': -zw, 'u': xw, 'v': yw, 'hu': dims.x*0.5, 'hv': dims.y*0.5, 'thickness': dims.z, 'dir_in':  zw},
    }
    return frames

class SBX_OT_section_lines(Operator):
    bl_idname = 'sbx.section_lines'
    bl_label = 'Generate Box Section (selected face)'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        box = _get_active_box(context)
        if not box:
            self.report({'WARNING'}, 'Select a SectionBox first')
            return {'CANCELLED'}
        targets = _get_targets_from_box(box)
        if not targets:
            self.report({'WARNING'}, 'No targets bound to this SectionBox')
            return {'CANCELLED'}
        s = context.scene.sbx_settings
        frames = _box_face_frames(box)
        face = frames.get(s.face_choice)
        if not face:
            self.report({'WARNING'}, 'Invalid face selection'); return {'CANCELLED'}
        col_main, col_cut, col_mesh, col_proj, idx, iter_tag = _next_iteration_collections_box(box, s.face_choice)
        diag = box.dimensions.length; eps = max(1e-6, diag * 1e-5)
        made_cut = made_proj = made_faces = 0
        for t in targets:
            # CUT (slightly inside)
            plane_point = face['center'] + face['dir_in'] * eps
            cut_polylines = _intersect_mesh_with_plane(t, plane_point, face['normal'])
            cut_segments = []
            for line in cut_polylines:
                for i in range(len(line)-1):
                    p1, p2 = line[i], line[i+1]
                    u1, v1 = _project_to_frame(p1, face)
                    u2, v2 = _project_to_frame(p2, face)
                    c = _clip_segment_to_rect(u1, v1, u2, v2, face['hu'], face['hv'])
                    if not c: continue
                    cu1, cv1, cu2, cv2 = c
                    wp1 = _unproject_from_frame(cu1, cv1, face)
                    wp2 = _unproject_from_frame(cu2, cv2, face)
                    cut_segments.append((wp1, wp2))
            stitched_cut = _stitch_segments(cut_segments, tol=1e-6)
            for pts in stitched_cut:
                if len(pts) < 2: continue
                cu = _make_curve_from_polyline(pts, name=f'SBX_Line_cut_{iter_tag}_{t.name}')
                _link_exclusive_to_collection(cu, col_cut); made_cut += 1
                closed = _ensure_closed(pts, tol=1e-5)
                if len(closed) >= 4 and (closed[0]-closed[-1]).length <= 1e-4:
                    face_obj = _make_ngon_face(closed, name=f'SBX_FillFace_{iter_tag}_{t.name}')
                    if face_obj:
                        _link_exclusive_to_collection(face_obj, col_mesh); made_faces += 1
            # PROJECTION (slab)
            thickness = face['thickness']; tmin = eps; tmax = max(eps, thickness - eps)
            edge_segments = _all_mesh_edges_world(t)
            proj_segments = []
            for (a, b) in edge_segments:
                clipped = _clip_edge_to_slab(a, b, face, tmin, tmax)
                if not clipped: continue
                qa, qb = clipped
                pa = _project_point_onto_plane(qa, face)
                pb = _project_point_onto_plane(qb, face)
                u1, v1 = _project_to_frame(pa, face)
                u2, v2 = _project_to_frame(pb, face)
                c = _clip_segment_to_rect(u1, v1, u2, v2, face['hu'], face['hv'])
                if not c: continue
                cu1, cv1, cu2, cv2 = c
                wp1 = _unproject_from_frame(cu1, cv1, face)
                wp2 = _unproject_from_frame(cu2, cv2, face)
                proj_segments.append((wp1, wp2))
            stitched_proj = _stitch_segments(proj_segments, tol=1e-6)
            for pts in stitched_proj:
                if len(pts) < 2: continue
                cu = _make_curve_from_polyline(pts, name=f'SBX_Line_projection_{iter_tag}_{t.name}')
                _link_exclusive_to_collection(cu, col_proj); made_proj += 1
        self.report({'INFO'}, f'Box S{idx:03d} {s.face_choice}: {made_cut} cut, {made_proj} proj, {made_faces} faces')
        return {'FINISHED'}

# ------------------------------- PLANE logic --------------------------------

def _ensure_planes_root():
    name = "SBP_Planes"
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

class SBP_OT_create_plane(Operator):
    bl_idname = 'sbp.create_plane'
    bl_label = 'Create Cutting Plane'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        s = context.scene.sbx_settings
        bpy.ops.mesh.primitive_plane_add(location=(0,0,0))
        pl = context.active_object
        pl.name = PLANE_PREFIX
        _ensure_plane_display(pl)
        # Orient:
        if s.plane_orient == 'XY':
            pl.rotation_euler = (0.0, 0.0, 0.0)            # normal +Z
        elif s.plane_orient == 'XZ':
            pl.rotation_euler = (0.0, 1.57079632679, 0.0)  # normal +Y
        else:  # YZ
            pl.rotation_euler = (1.57079632679, 0.0, 0.0)  # normal +X
        # Apply initial size via anchors (center by default)
        _apply_plane_sizes_with_anchors(pl, s.plane_size_u, s.plane_size_v, s.plane_anchor_u, s.plane_anchor_v)
        # Move to dedicated planes root
        root = _ensure_planes_root()
        _link_exclusive_to_collection(pl, root)
        # Sync UI to actual dims
        s.plane_size_u, s.plane_size_v = pl.dimensions.x, pl.dimensions.y
        # Init empty target list
        _set_targets_prop_plane(pl, [])
        self.report({'INFO'}, f'Created plane: {s.plane_orient}')
        return {'FINISHED'}

class SBP_OT_offset_plane(Operator):
    bl_idname = 'sbp.offset_plane'
    bl_label = 'Offset Active Plane'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        pl = _get_active_plane(context)
        if not pl:
            self.report({'WARNING'}, 'Select a SectionPlane first')
            return {'CANCELLED'}
        s = context.scene.sbx_settings
        frame = _plane_frame_from_object(pl)
        pl.location += frame['normal'] * s.plane_offset
        s.plane_offset = 0.0
        return {'FINISHED'}

class SBP_OT_bind_targets(Operator):
    bl_idname = 'sbp.bind_targets'
    bl_label = 'Bind Selection to Active Plane'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        pl = _get_active_plane(context)
        if not pl:
            self.report({'WARNING'}, 'Select a SectionPlane first')
            return {'CANCELLED'}
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if not meshes:
            self.report({'WARNING'}, 'Select mesh objects to bind')
            return {'CANCELLED'}
        _set_targets_prop_plane(pl, meshes)
        self.report({'INFO'}, f'Bound {len(meshes)} mesh(es) to plane')
        return {'FINISHED'}

def _clip_edge_to_plane_depth(a: Vector, b: Vector, frame, tmin: float, tmax: float, dir_sign: float):
    d = b - a
    dir_vec = frame['normal'] * dir_sign
    ta = (a - frame['center']).dot(dir_vec)
    tb = (b - frame['center']).dot(dir_vec)
    if (ta < tmin and tb < tmin) or (ta > tmax and tb > tmax): return None
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
    if (qb - qa).length <= 1e-9: return None
    return (qa, qb)

class SBP_OT_generate(Operator):
    bl_idname = 'sbp.generate'
    bl_label = 'Generate Planar Section'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        pl = _get_active_plane(context)
        if not pl:
            self.report({'WARNING'}, 'Select a SectionPlane first')
            return {'CANCELLED'}
        targets = _get_targets_from_plane(pl)
        if not targets:
            targets = [o for o in context.selected_objects if o.type == 'MESH']
            if not targets:
                self.report({'WARNING'}, 'No targets bound/selected for planar section')
                return {'CANCELLED'}
        s = context.scene.sbx_settings
        frame = _plane_frame_from_object(pl)
        face_tag = f"{s.plane_orient}{'P' if s.plane_dir=='POS' else 'N'}"
        col_main, col_plane, col_cut, col_mesh, col_elev, idx, iter_tag = _next_iteration_collections_plane(pl, face_tag)
        # Freeze a copy of the plane in the iteration
        pl_copy_me = pl.data.copy()
        pl_copy = bpy.data.objects.new(f"SBP_Plane_{iter_tag}", pl_copy_me)
        pl_copy.matrix_world = pl.matrix_world.copy()
        _link_exclusive_to_collection(pl_copy, col_plane)
        dir_sign = 1.0 if s.plane_dir == 'POS' else -1.0
        depth = max(0.0, s.plane_depth)
        eps = max(1e-6, 1e-5 * (s.plane_size_u + s.plane_size_v))
        made_cut = made_elev = made_faces = 0
        for t in targets:
            plane_point = frame['center'] + frame['normal'] * (eps * dir_sign * 0.5)
            cut_polylines = _intersect_mesh_with_plane(t, plane_point, frame['normal'])
            cut_segments = []
            for line in cut_polylines:
                for i in range(len(line)-1):
                    p1, p2 = line[i], line[i+1]
                    u1, v1 = _project_to_frame(p1, frame)
                    u2, v2 = _project_to_frame(p2, frame)
                    c = _clip_segment_to_rect(u1, v1, u2, v2, frame['hu'], frame['hv'])
                    if not c: continue
                    cu1, cv1, cu2, cv2 = c
                    wp1 = _unproject_from_frame(cu1, cv1, frame)
                    wp2 = _unproject_from_frame(cu2, cv2, frame)
                    cut_segments.append((wp1, wp2))
            stitched_cut = _stitch_segments(cut_segments, tol=1e-6)
            for pts in stitched_cut:
                if len(pts) < 2: continue
                cu = _make_curve_from_polyline(pts, name=f"SBP_Line_cut_{iter_tag}_{t.name}")
                _link_exclusive_to_collection(cu, col_cut); made_cut += 1
                closed = _ensure_closed(pts, tol=1e-5)
                if len(closed) >= 4 and (closed[0]-closed[-1]).length <= 1e-4:
                    face_obj = _make_ngon_face(closed, name=f"SBP_FillFace_{iter_tag}_{t.name}")
                    if face_obj:
                        _link_exclusive_to_collection(face_obj, col_mesh); made_faces += 1
            # Elevation (projection in slab)
            edge_segments = _all_mesh_edges_world(t)
            elev_segments = []
            tmin = eps
            tmax = max(eps, depth) if depth > 0.0 else eps
            if tmax > tmin + 1e-9:
                for (a, b) in edge_segments:
                    clipped = _clip_edge_to_plane_depth(a, b, frame, tmin, tmax, dir_sign=dir_sign)
                    if not clipped: continue
                    qa, qb = clipped
                    pa = _project_point_onto_plane(qa, frame)
                    pb = _project_point_onto_plane(qb, frame)
                    u1, v1 = _project_to_frame(pa, frame)
                    u2, v2 = _project_to_frame(pb, frame)
                    c = _clip_segment_to_rect(u1, v1, u2, v2, frame['hu'], frame['hv'])
                    if not c: continue
                    cu1, cv1, cu2, cv2 = c
                    wp1 = _unproject_from_frame(cu1, cv1, frame)
                    wp2 = _unproject_from_frame(cu2, cv2, frame)
                    elev_segments.append((wp1, wp2))
            stitched_elev = _stitch_segments(elev_segments, tol=1e-6)
            for pts in stitched_elev:
                if len(pts) < 2: continue
                cu = _make_curve_from_polyline(pts, name=f"SBP_Line_elev_{iter_tag}_{t.name}")
                _link_exclusive_to_collection(cu, col_elev); made_elev += 1
        self.report({'INFO'}, f'Plane S{idx:03d} {face_tag}: {made_cut} cut, {made_elev} elev, {made_faces} faces')
        return {'FINISHED'}

# ------------------------------ SVG / DXF Export -----------------------------

def _normalize_export_path(path, ext, default_name):
    if not path:
        return bpy.path.abspath(f"//{default_name}.{ext}")
    p = bpy.path.abspath(path)
    # If path is a directory or ends with slash, join default file name
    if p.endswith(('/', '\\')) or os.path.isdir(p):
        p = os.path.join(p, f"{default_name}.{ext}")
    dot_ext = f".{ext}".lower()
    if not p.lower().endswith(dot_ext):
        # Ensure extension
        if p.endswith('.'):
            p = p + ext
        else:
            p = p + dot_ext
    dirpath = os.path.dirname(p) or bpy.path.abspath("//")
    os.makedirs(dirpath, exist_ok=True)
    return p

def _project_xy(p: Vector):
    return (p.x, p.y)

def _collect_world_polylines_from_collection(col: bpy.types.Collection, include_kinds=('cut','projection','elevation')):
    # kinds mapped by substring in object name
    want_cut = 'cut' in include_kinds
    want_proj = 'projection' in include_kinds
    want_elev = 'elevation' in include_kinds
    objs = []
    _walk_collection_objects(col, objs)
    out = []
    for ob in objs:
        if ob.type != 'CURVE':
            continue
        nm = ob.name
        if not (nm.startswith('SBX_Line_') or nm.startswith('SBP_Line_')):
            continue
        if ('_cut_' in nm and not want_cut): continue
        if ('_projection_' in nm and not want_proj): continue
        if ('_elev_' in nm and not want_elev): continue
        mw = ob.matrix_world.copy()
        for sp in ob.data.splines:
            if sp.type != 'POLY' or len(sp.points) < 2:
                continue
            pts = [mw @ Vector((p.co.x, p.co.y, p.co.z)) for p in sp.points]
            out.append(pts)
    return out

def _collect_fillfaces_world_polygons_from_collection(col: bpy.types.Collection):
    objs = []
    _walk_collection_objects(col, objs)
    polys = []
    for ob in objs:
        if ob.type != 'MESH':
            continue
        nm = ob.name
        if not (nm.startswith('SBX_FillFace_') or nm.startswith('SBP_FillFace_')):
            continue
        mw = ob.matrix_world; me = ob.data
        me.calc_loop_triangles()
        for poly in me.polygons:
            verts = [mw @ me.vertices[i].co for i in poly.vertices]
            if len(verts) >= 3:
                polys.append(verts)
    return polys

def _export_svg_from_collection(filepath, col: bpy.types.Collection):
    cut_lines  = _collect_world_polylines_from_collection(col, include_kinds=('cut',))
    proj_lines = _collect_world_polylines_from_collection(col, include_kinds=('projection',))
    elev_lines = _collect_world_polylines_from_collection(col, include_kinds=('elevation',))
    fills      = _collect_fillfaces_world_polygons_from_collection(col)

    if not (cut_lines or proj_lines or elev_lines or fills):
        return False, 'No outputs found in the last iteration collection'

    # Project all to XY (2D export)
    def proj_all(seq_of_polylines):
        return [[_project_xy(p) for p in pl] for pl in seq_of_polylines]

    cut2  = proj_all(cut_lines)
    proj2 = proj_all(proj_lines)
    elev2 = proj_all(elev_lines)
    fills2 = proj_all(fills)

    xs = [xy[0] for pl in (cut2 + proj2 + elev2 + fills2) for xy in pl]
    ys = [xy[1] for pl in (cut2 + proj2 + elev2 + fills2) for xy in pl]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    w = max(1.0, maxx - minx); h = max(1.0, maxy - miny)

    svg = ET.Element('svg', attrib={'xmlns':'http://www.w3.org/2000/svg','width':str(w),'height':str(h),'viewBox':f'0 0 {w} {h}'})
    if fills2:
        g_fills = ET.SubElement(svg, 'g', attrib={'id':'fills'})
        for pl in fills2:
            pts = []
            for (x, y) in pl:
                sx = x - minx; sy = (maxy - y)
                pts.append(f'{sx},{sy}')
            ET.SubElement(g_fills, 'polygon', attrib={'points':' '.join(pts),'fill':'#b3b3b3','stroke':'none'})
    stroke_w = '0.25'
    if cut2:
        g_cut = ET.SubElement(svg, 'g', attrib={'id':'cut'})
        for pl in cut2:
            parts = []
            for (x, y) in pl:
                sx = x - minx; sy = (maxy - y)
                parts.append(f'{sx},{sy}')
            ET.SubElement(g_cut, 'polyline', attrib={'points':' '.join(parts),'fill':'none','stroke':'black','stroke-width':stroke_w})
    if proj2:
        g_proj = ET.SubElement(svg, 'g', attrib={'id':'projection'})
        for pl in proj2:
            parts = []
            for (x, y) in pl:
                sx = x - minx; sy = (maxy - y)
                parts.append(f'{sx},{sy}')
            ET.SubElement(g_proj, 'polyline', attrib={'points':' '.join(parts),'fill':'none','stroke':'black','stroke-width':stroke_w})
    if elev2:
        g_elev = ET.SubElement(svg, 'g', attrib={'id':'elevation'})
        for pl in elev2:
            parts = []
            for (x, y) in pl:
                sx = x - minx; sy = (maxy - y)
                parts.append(f'{sx},{sy}')
            ET.SubElement(g_elev, 'polyline', attrib={'points':' '.join(parts),'fill':'none','stroke':'black','stroke-width':stroke_w})

    ET.ElementTree(svg).write(filepath, encoding='utf-8', xml_declaration=False)
    return True, f'SVG saved: {filepath}'

def _export_dxf_from_collection(filepath, col: bpy.types.Collection):
    polylines = _collect_world_polylines_from_collection(col, include_kinds=('cut','projection','elevation'))
    if not polylines:
        return False, 'No line outputs found in the last iteration collection'

    # Project all to XY (2D DXF)
    polylines2 = [[_project_xy(p) for p in pl] for pl in polylines]

    def w(f, s): f.write(s); f.write("\n")
    with open(filepath, 'w', encoding='utf-8') as f:
        w(f,'0'); w(f,'SECTION'); w(f,'2'); w(f,'ENTITIES')
        for pl in polylines2:
            for i in range(len(pl)-1):
                x1,y1 = pl[i]; x2,y2 = pl[i+1]
                w(f,'0'); w(f,'LINE')
                w(f,'8'); w(f,'SECTION')  # layer
                w(f,'10'); w(f,str(float(x1))); w(f,'20'); w(f,str(float(y1))); w(f,'30'); w(f,'0.0')
                w(f,'11'); w(f,str(float(x2))); w(f,'21'); w(f,str(float(y2))); w(f,'31'); w(f,'0.0')
        w(f,'0'); w(f,'ENDSEC'); w(f,'0'); w(f,'EOF')
    return True, f'DXF saved: {filepath}'

def _export_last_from_box(base_path, to_svg=True):
    box = _get_active_box(bpy.context)
    if not box:
        return False, "Select a SectionBox to export its last iteration"
    latest, idx = _find_latest_iteration_box(box)
    if not latest:
        return False, "No iterations found for this SectionBox"
    # derive tag (S###_FACE)
    tag = latest.name.split("__Section_")[-1]
    ext = 'svg' if to_svg else 'dxf'
    path = _normalize_export_path(base_path, ext, f"SBX_{box.name}_{tag}")
    if to_svg:
        return _export_svg_from_collection(path, latest)
    else:
        return _export_dxf_from_collection(path, latest)

def _export_last_from_plane(base_path, to_svg=True):
    pl = _get_active_plane(bpy.context)
    if not pl:
        return False, "Select a SectionPlane to export its last iteration"
    latest, idx = _find_latest_iteration_plane(pl)
    if not latest:
        return False, "No iterations found for this SectionPlane"
    tag = latest.name.split("__Section_")[-1]
    ext = 'svg' if to_svg else 'dxf'
    path = _normalize_export_path(base_path, ext, f"SBP_{pl.name}_{tag}")
    if to_svg:
        return _export_svg_from_collection(path, latest)
    else:
        return _export_dxf_from_collection(path, latest)

class SBX_OT_export_box_svg(Operator):
    bl_idname = 'sbx.export_box_svg'
    bl_label = 'Export SVG (last Box run)'
    bl_options = {'REGISTER'}
    def execute(self, context):
        s = context.scene.sbx_settings
        ok, msg = _export_last_from_box(s.export_path, to_svg=True)
        self.report({'INFO' if ok else 'WARNING'}, msg)
        return {'FINISHED'}

class SBX_OT_export_box_dxf(Operator):
    bl_idname = 'sbx.export_box_dxf'
    bl_label = 'Export DXF (last Box run)'
    bl_options = {'REGISTER'}
    def execute(self, context):
        s = context.scene.sbx_settings
        ok, msg = _export_last_from_box(s.export_path, to_svg=False)
        self.report({'INFO' if ok else 'WARNING'}, msg)
        return {'FINISHED'}

class SBP_OT_export_plane_svg(Operator):
    bl_idname = 'sbp.export_plane_svg'
    bl_label = 'Export SVG (last Plane run)'
    bl_options = {'REGISTER'}
    def execute(self, context):
        s = context.scene.sbx_settings
        ok, msg = _export_last_from_plane(s.export_path, to_svg=True)
        self.report({'INFO' if ok else 'WARNING'}, msg)
        return {'FINISHED'}

class SBP_OT_export_plane_dxf(Operator):
    bl_idname = 'sbp.export_plane_dxf'
    bl_label = 'Export DXF (last Plane run)'
    bl_options = {'REGISTER'}
    def execute(self, context):
        s = context.scene.sbx_settings
        ok, msg = _export_last_from_plane(s.export_path, to_svg=False)
        self.report({'INFO' if ok else 'WARNING'}, msg)
        return {'FINISHED'}

# ----------------------------------- UI -------------------------------------

class SBX_PT_panel(Panel):
    bl_label = 'Section Toolbox'
    bl_idname = 'SBX_PT_panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Section Toolbox'

    def draw(self, context):
        s = context.scene.sbx_settings
        layout = self.layout

        # --- Box creation & params
        col = layout.column(align=True)
        col.label(text='Create Section Box')
        col.operator('sbx.create_from_selection')

        col.separator()
        col.label(text='Parametric Box')
        col.prop(s, 'live_link')
        row = col.row(align=True); row.prop(s, 'width'); row.prop(s, 'depth'); row.prop(s, 'height')
        row = col.row(align=True); row.prop(s, 'anchor_x'); row.prop(s, 'anchor_y'); row.prop(s, 'anchor_z')

        col.separator()
        col.label(text='Move')
        row = col.row(align=True); row.prop(s, 'move_x'); row.prop(s, 'move_y'); row.prop(s, 'move_z')

        col.separator()
        col.label(text='Rotate (degrees)')
        row = col.row(align=True); row.prop(s, 'rot_x'); row.prop(s, 'rot_y'); row.prop(s, 'rot_z')

        col.separator()
        col.label(text='Bind / Targets (Box)')
        row = col.row(align=True)
        op = row.operator('sbx.bind_targets', text='Bind Selection'); op.mode = 'SELECTION'
        op = row.operator('sbx.bind_targets', text='Bind Collection'); op.mode = 'COLLECTION'
        col.prop(s, 'collection_name', text='Collection')

        col.separator()
        col.label(text='Live Boolean Control')
        row = col.row(align=True)
        op = row.operator('sbx.toggle_live', text='Enable Live');  op.enable = True
        op = row.operator('sbx.toggle_live', text='Disable Live'); op.enable = False

        col.separator()
        col.label(text='Box Sections (pick a face)')
        col.prop(s, 'face_choice')
        col.operator('sbx.section_lines', text='Generate Box Section (versioned)')

        # --- Planar subset
        layout.separator()
        box_ui = layout.box()
        box_ui.label(text='Planar Sections')
        box_ui.prop(s, 'plane_live_link', text='Plane Live Link')
        row = box_ui.row(align=True)
        row.prop(s, 'plane_orient'); row.prop(s, 'plane_dir'); row.prop(s, 'plane_depth')
        row2 = box_ui.row(align=True)
        row2.prop(s, 'plane_size_u'); row2.prop(s, 'plane_size_v')
        row3 = box_ui.row(align=True)
        row3.prop(s, 'plane_anchor_u'); row3.prop(s, 'plane_anchor_v')
        row4 = box_ui.row(align=True)
        row4.operator('sbp.create_plane', text='Create Plane')
        row4.prop(s, 'plane_offset'); row4.operator('sbp.offset_plane', text='Apply Offset')
        row5 = box_ui.row(align=True)
        row5.operator('sbp.bind_targets', text='Bind Selection to Plane')
        row6 = box_ui.row(align=True)
        row6.operator('sbp.generate', text='Generate Planar Section (versioned)')

        # --- Export (per context)
        layout.separator()
        ex = layout.box()
        ex.label(text='Export (last iteration)')
        ex.prop(s, 'export_path')
        r = ex.row(align=True)
        r.operator('sbx.export_box_svg'); r.operator('sbx.export_box_dxf')
        r = ex.row(align=True)
        r.operator('sbp.export_plane_svg'); r.operator('sbp.export_plane_dxf')

        # --- Finalize
        layout.separator()
        col = layout.column(align=True)
        col.label(text='Finalize')
        col.operator('sbx.apply', text='Bake / Apply')
        col.operator('sbx.cleanup', text='Remove Box + Mods', icon='TRASH')

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
    if hasattr(bpy.types.Scene, 'sbx_settings'):
        del bpy.types.Scene.sbx_settings

if __name__ == '__main__':
    register()

