import bpy, os, re, sys, numpy as np
import bmesh
from mathutils.kdtree import KDTree
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from collections import Counter

# ---------- helpers for labels ----------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--")+1:]
    else:
        raise SystemExit("Missing -- separator. Run blender ... --python script.py -- <obj> <target_faces> <out.obj> [labels...]")

    if len(argv) < 3:
        raise SystemExit("Usage: ... -- <obj> <target_faces> <out.obj> [label1 ...]")

    obj_file = argv[0]
    target_faces = int(argv[1])
    export_name = argv[2]
    label_paths = argv[3:]                # <- all extras
    return obj_file, target_faces, export_name, label_paths

def label_basename(p):
    # "…/1/1_0.seg" -> "1_0"
    return os.path.splitext(os.path.basename(p))[0]

def make_attr_name(base):
    # keep underscores, keep it Blender-safe
    import re
    name = re.sub(r'[^A-Za-z0-9_]', '_', base)
    if not re.match(r'[A-Za-z_]', name):
        name = 'seg_' + name
    return name[:60]


def ensure_face_label_from_point(mesh, *, point_attr_name="label_0", face_attr_name="face_label"):
    """
    If FACE attribute `face_label` is missing, derive it from POINT attribute `label_0`
    by majority vote over the three vertices of each face.
    Returns the name of the face attribute if created/found, else None.
    """
    # already there?
    if mesh.attributes.get(face_attr_name) is not None:
        return face_attr_name

    pa = mesh.attributes.get(point_attr_name)
    if pa is None:
        print(f"[WARN] POINT attribute '{point_attr_name}' not found; cannot build '{face_attr_name}'.")
        return None

    # read per-vertex ints
    vvals = np.fromiter((d.value for d in pa.data), count=len(pa.data), dtype=np.int32)

    # majority per face
    fvals = np.empty(len(mesh.polygons), dtype=np.int32)
    for i, poly in enumerate(mesh.polygons):
        vids = poly.vertices
        a, b, c = vvals[vids[0]], vvals[vids[1]], vvals[vids[2]]
        # majority (tie -> min label)
        if a == b or a == c:
            f = a
        elif b == c:
            f = b
        else:
            f = min(a, b, c)
        fvals[i] = f

    fa = mesh.attributes.new(face_attr_name, type='INT', domain='FACE')
    # foreach_set for INT works on .value
    fa.data.foreach_set("value", fvals.tolist())
    print(f"[INFO] Created FACE attribute '{face_attr_name}' from POINT '{point_attr_name}'.")
    return face_attr_name


def write_face_int_attribute(mesh, name, values):
    # mesh: bpy.types.Mesh; values: len == len(mesh.polygons)
    if name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[name])
    attr = mesh.attributes.new(name=name, type='INT', domain='FACE')
    a = mesh.attributes[name].data
    for i, v in enumerate(values):
        a[i].value = int(v)

def save_face_int_attributes_npz(mesh, names, npz_path):
    out = {}
    for n in names:
        if n not in mesh.attributes or mesh.attributes[n].domain != 'FACE':
            continue
        out[n] = np.array([mesh.attributes[n].data[i].value
                           for i in range(len(mesh.polygons))], dtype=np.int32)
    if out:
        np.savez(npz_path, **out)

def transfer_face_labels_nearest_face(src_obj, dst_obj, face_attr_names):
    """Copy each FACE int attribute by nearest-face lookup (centroid -> BVH)."""
    me_s, me_d = src_obj.data, dst_obj.data

    # build BVH on source faces
    verts_s = [v.co.copy() for v in me_s.vertices]
    polys_s = [list(p.vertices) for p in me_s.polygons]
    bvh = BVHTree.FromPolygons(verts_s, polys_s)

    # precompute dst centroids -> nearest source face id
    nearest_src_f = []
    for p in me_d.polygons:
        c = sum((me_d.vertices[vi].co for vi in p.vertices), Vector()) / len(p.vertices)
        hit = bvh.find_nearest(c)
        nearest_src_f.append(hit[2] if hit and hit[2] is not None else None)

    # transfer each requested face attribute
    for name in face_attr_names:
        if name not in me_s.attributes or me_s.attributes[name].domain != 'FACE':
            continue
        src_vals = me_s.attributes[name].data
        dst_vals = np.zeros(len(me_d.polygons), dtype=np.int32)
        for i, si in enumerate(nearest_src_f):
            dst_vals[i] = int(src_vals[si].value) if si is not None else 0
        write_face_int_attribute(me_d, name, dst_vals)


def ensure_boundary_vgroup(obj, face_attr_name="face_label", group_name="KEEP_BOUNDARY", ring=0):
    me = obj.data
    vg = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)

    # read face labels
    lab = [0]*len(me.polygons)
    if face_attr_name in me.attributes and me.attributes[face_attr_name].domain == 'FACE':
        data = me.attributes[face_attr_name].data
        for i in range(len(me.polygons)):
            lab[i] = int(data[i].value)
    else:
        raise RuntimeError(f"Missing face attribute {face_attr_name}")

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    boundary_verts = set()
    for e in bm.edges:
        fs = [f.index for f in e.link_faces]
        if len(fs) == 2 and lab[fs[0]] != lab[fs[1]]:
            boundary_verts.add(e.verts[0].index)
            boundary_verts.add(e.verts[1].index)

    # optional: expand a few rings
    if ring > 0 and boundary_verts:
        # build 1-step vertex adjacency via edges
        nbr = {}
        for v in bm.verts:
            adj = set()
            for e in v.link_edges:
                adj.add(e.other_vert(v).index)
            nbr[v.index] = adj

        front = set(boundary_verts)
        for _ in range(ring):
            nxt = set()
            for vi in front:
                nxt |= nbr.get(vi, set())
            boundary_verts |= nxt
            front = nxt

    bm.free()

    # assign weights: 1.0 on boundary verts
    vg.add(list(boundary_verts), 1.0, 'REPLACE')
    return vg

# ---------- your original process, adapted ----------

class Process:
    def __init__(self, obj_file, target_faces, export_name, label_paths):
        src = self.load_obj(obj_file)

        nF_src = len(src.data.polygons)
        print(f"[SANITY] Source faces: {nF_src}")

        def _natural_key(s):
            return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(s))]
        label_paths = sorted(label_paths, key=_natural_key)

        # load per-face labels from .seg/.ser (one int per line)
        def load_label_specs(paths):
            arrs = []
            for p in paths:
                a = np.loadtxt(p, dtype=np.int32)
                # ensure 1D
                a = np.asarray(a).reshape(-1)
                arrs.append(a)
            return arrs

        f_label_arrays = load_label_specs(label_paths)

        # 1) attach as FACE attributes on the source
        me_src = src.data
        face_attr_names = []
        for p, arr in zip(label_paths, f_label_arrays):
            name = make_attr_name(label_basename(p))  # e.g., "1_0"
            write_face_int_attribute(me_src, name, arr)
            face_attr_names.append(name)

        # 2) duplicate, simplify the duplicate only
        dst = src.copy()
        dst.data = src.data.copy()        # copies FACE attributes too
        dst.name = src.name + "_decimated"
        bpy.context.collection.objects.link(dst)

        # choose which FACE label defines boundaries (use the first one)
        boundary_face_attr = face_attr_names[0]

        # 3) simplify (protect boundaries of that face label)
        self.simplify(dst, target_faces, protect_boundaries=True, boundary_face_attr=boundary_face_attr)

        # 4) refresh labels on the simplified mesh via nearest-face transfer
        transfer_face_labels_nearest_face(src, dst, face_attr_names)

        # 5) export + save labels
        self.export_obj(dst, export_name)
        npz_face = os.path.splitext(export_name)[0] + "_face_labels.npz"
        save_face_int_attributes_npz(dst.data, face_attr_names, npz_face)
        print("Saved face labels:", npz_face)

    def load_obj(self, obj_file):
        bpy.ops.wm.obj_import(
            filepath=obj_file,
            forward_axis='NEGATIVE_Z',
            up_axis='Y',
            use_split_objects=False,
            use_split_groups=False
        )
        ob = bpy.context.selected_objects[0]
        # Force the Blender object name to match the file stem
        ob.name = os.path.splitext(os.path.basename(obj_file))[0]
        return ob

    def subsurf(self, mesh):
        bpy.context.view_layer.objects.active = mesh
        mod = mesh.modifiers.new(name='Subsurf', type='SUBSURF')
        mod.subdivision_type = 'SIMPLE'
        bpy.ops.object.modifier_apply(modifier=mod.name)
        mod = mesh.modifiers.new(name='Triangulate', type='TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # def simplify(self, mesh, target_faces, protect_boundaries=True, boundary_face_attr=None):
    #     bpy.context.view_layer.objects.active = mesh
    #     mod = mesh.modifiers.new(name='Decimate', type='DECIMATE')
    #     mod.use_collapse_triangulate = True

    #     if protect_boundaries and boundary_face_attr:
    #         vg = ensure_boundary_vgroup(mesh, face_attr_name=boundary_face_attr,
    #                                     group_name="KEEP_BOUNDARY", ring=1)
    #         if vg is not None:
    #             mod.vertex_group = vg.name  # bias collapse away from boundary verts
    #         else:
    #             print(f"[WARN] Boundary vgroup not created (missing '{boundary_face_attr}'). Proceeding without protection.")

    #     nfaces = len(mesh.data.polygons)
    #     if nfaces < target_faces:
    #         self.subsurf(mesh); nfaces = len(mesh.data.polygons)
    #     mod.ratio = float(target_faces) / float(nfaces)
    #     bpy.ops.object.modifier_apply(modifier=mod.name)
    def simplify(self, obj, target_faces, *,
            protect_boundaries=True,
            boundary_face_attr="face_label",
            boundary_ring=1,
            vg_factor=0.0,          # < 1.0 = softer protection; 0 disables the weighting effect
            tol=0.05,               # accept within ±5% of target
            max_passes=3):
        """
        Decimate to ~target_faces (counted as TRIANGLES).
        We triangulate *before* decimation so counts align with export.
        If boundary protection is too strong, reduce vg_factor or set protect_boundaries=False.
        """
        # vg_factor 1.0 to remove collision
        # boundary ring protects a band of vertices around the seam
        bpy.context.view_layer.objects.active = obj

        # 0) Triangulate *first* so counts are in triangles (matches export)
        tri_pre = obj.modifiers.new(name='TriangulatePre', type='TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier=tri_pre.name)

        # 1) Optional boundary protection
        vgname = ""
        if protect_boundaries:
            vg = ensure_boundary_vgroup(obj,
                                        face_attr_name=boundary_face_attr,
                                        group_name="KEEP_BOUNDARY",
                                        ring=boundary_ring)
            vgname = vg.name

        def tri_face_count():
            return len(obj.data.polygons)

        n0 = tri_face_count()
        if n0 <= target_faces:
            # Already at or below target (in triangles)
            return

        # Initial ratio guess
        ratio = float(target_faces) / float(n0)

        for pass_idx in range(max_passes):
            dec = obj.modifiers.new(name=f'Decimate_{pass_idx}', type='DECIMATE')
            dec.use_collapse_triangulate = True
            dec.ratio = max(min(ratio, 1.0), 0.001)

            if vgname:
                dec.vertex_group = vgname
                # Soften protection: 1.0 = very strong; 0.5 = moderate; 0.0 = no effect
                dec.vertex_group_factor = float(vg_factor)

            bpy.ops.object.modifier_apply(modifier=dec.name)

            ncur = tri_face_count()
            print(f"[Decimate pass {pass_idx}] faces: {n0} -> {ncur} (target {target_faces}, ratio {ratio:.6f})")

            # close enough?
            if ncur <= target_faces * (1.0 + tol):
                break

            # Update ratio for next pass based on current count (feedback control)
            if ncur > 0:
                ratio *= float(target_faces) / float(ncur)
            else:
                break

        # 2) Ensure final mesh is triangulated (safety, should already be)
        tri_post = obj.modifiers.new(name='TriangulatePost', type='TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier=tri_post.name)

    def export_obj(self, mesh, export_name):
        outdir = os.path.dirname(export_name)
        if outdir and not os.path.isdir(outdir): os.makedirs(outdir)
        print('EXPORTING', export_name)
        bpy.ops.object.select_all(action='DESELECT')
        mesh.select_set(state=True)
        bpy.ops.wm.obj_export(
            filepath=export_name, check_existing=False, export_selected_objects=True,
            global_scale=1, path_mode='AUTO',
            forward_axis='NEGATIVE_Z', up_axis='Y',
            export_eval_mode='DAG_EVAL_VIEWPORT',
            export_uv=False, export_normals=True, export_materials=False,
            export_triangulated_mesh=True,
            export_object_groups=False, export_smooth_groups=False, export_vertex_groups=False
        )

# ---- CLI ----
# usage:
# blender --background --python this_script.py -- <obj> <target_faces> <out.obj> <label1.npy> <label2.npy> ...
# argv = sys.argv
# sep = argv.index('--') if '--' in argv else len(argv)
# obj_file = argv[sep+1]
# target_faces = int(argv[sep+2])
# export_name = argv[sep+3]
# label_paths = argv[sep+4:]        # 10–13 files

# print('args:', obj_file, target_faces, export_name, f'labels={len(label_paths)}')
# Process(obj_file, target_faces, export_name, label_paths)

obj_file, target_faces, export_name, label_paths = parse_args()
print("OBJ:", obj_file)
print("Target faces:", target_faces)
print("Out:", export_name)
print("Labels ({}):".format(len(label_paths)))
for lp in label_paths:
    print("  -", os.path.basename(lp))
# for lp in label_paths: print("  -", os.path.basename(lp))

# If you also derive attribute names:
attr_names = [make_attr_name(label_basename(p)) for p in label_paths]
print("Face attributes to create:", attr_names)

blender = Process(obj_file, target_faces, export_name, label_paths)