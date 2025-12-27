import os
import trimesh
import open3d as o3d
import numpy as np

## Util related to io ##

def read_mesh(dir_path, only_pref=False, ext=(".off", ".ply", ".obj")):
    """_summary_

    Args:
        dir_path (str): directory path where meshes are
        supporting mesh file extensions: .obj, .stl, .ply, .off

    Returns:
        list of meshes (trimesh.Trimesh): list of meshes
        Trimesh
        ├── vertices [n, 3]
        ├── faces [m, 3]
        ├── vertex_normals [n, 3]
        ├── face_normals [m, 3]
        ├── edges / edges_unique
        ├── is_watertight
        ├── volume / area / centroid
        ├── methods: show(), export(), fill_holes(), split()
    """
    fnames = [f for f in os.listdir(dir_path) if f.endswith(ext)]
    # count = len(os.listdir(dir_path))
    # fnames = ["{}.off".format(f) for f in range(1, count+1)]
    # _key = [int(f.split(".")[0]) for f in fnames]
    sfnames = sorted(fnames, key=lambda f: int(f.split(".")[0]))
    meshes = []
    for f in sfnames:
        fpath = os.path.join(dir_path, f)
        print(fpath)
        mesh = trimesh.load(fpath)
        meshes.append(mesh)
    if only_pref:
        sfnames = ["".join([f.split(".")[0].split("_")[0], ".obj"]) for f in sfnames]

    return meshes, sfnames


# Read face segmentation
def read_seg_res(dir_path, layer=0):
    """Read segmentation result and seg(eseg)/sseg(seseg) file names
    Args:
        dir_path (python path): path where ground truth segmentation is saved
        layer (int, optional): nested file layer number. Defaults to 0.
    Returns:
        point_seg (python list): loaded segmentation in python list of lists. Inner component is ndarray
        sfile/sdir names: names of directory, files in sorted order
        seg_label_names (python list): doubly list of tags
    """
    if layer == 0:
        filenames = [d for d in os.listdir(dir_path)] # directory names
        sfilenames = sorted(filenames, key=lambda d: int(d.split(".")[0].split("_")[0])) # sorted in number
        seg_files = ["{}.npz".format(f.split(".")[0]) for f in sfilenames] # read face labels in for each directory
        face_labels = []
        seg_label_names = []
        for elem in seg_files:
            fpath = os.path.join(dir_path, elem) # mult seg for sing mesh
            part_label = np.load(fpath) # load labels
            part_label_tag = part_label.files # single mesh - mult face label
            point_seg = []
            seg_label_names.append(part_label_tag) # add face label
            for plabel_name in part_label_tag:
                point_seg.append(part_label[plabel_name])
            face_labels.append(point_seg) # add to point labels

        return face_labels, sfilenames, seg_label_names
    
    elif layer == 1:
        dirnames = [d for d in os.listdir(dir_path)] # class dir names
        sdirnames = sorted(dirnames, key=lambda d: int(d.split(".")[0].split("_")[0])) # sorted class
        face_labels = []
        for d in sdirnames: # d is class directory
            seg_path = os.path.join(dir_path, d) # path name
            count = len(os.listdir(seg_path)) # seg count for single mesh
            point_seg = []
            seg_res = ["{}_{}.seg".format(d,i) for i in range(count)] # single mesh - mult seg
            for elem in seg_res:
                fpath = os.path.join(seg_path, elem) # single label path
                part_label = np.loadtxt(fname=fpath, dtype=int) # load label
                point_seg.append(part_label) # append to face seg
            face_labels.append(point_seg)

        return face_labels, sdirnames


def read_seg_res_with_eseg_fname(dir_path, layer=0):
    """Read segmentation result and seg(eseg)/sseg(seseg) file names
    Args:
        dir_path (python path): path where ground truth segmentation is saved
        layer (int, optional): nested file layer number. Defaults to 0.

    Returns:
        face_seg (python list): loaded segmentation in python list of lists. Inner component is ndarray
        eseg_dirs (python list): segmentation name list of .eseg extension in python list of lists.
        seseg_dirs (python list): segmentation name list of .seseg extension in python list of lists.
    """
    if layer == 1:
        dirnames = [d for d in os.listdir(dir_path)] # dir names - classs
        sdirnames = sorted(dirnames, key=lambda d: int(d.split(".")[0].split("_")[0])) # sorted in int size
        face_labels = [] # labels for face (for all class)
        eseg_dirs = [] # hard label dir
        seseg_dirs = [] # soft label dir
        for d in sdirnames:
            seg_path = os.path.join(dir_path, d) # face label path
            count = len(os.listdir(seg_path)) # num segmentation
            face_seg = []
            seg_res = ["{}_{}.seg".format(d,i) for i in range(count)] # flabel
            eseg_name = ["{}_{}.eseg".format(d,i) for i in range(count)] # hlabel
            eseg_dirs.append(eseg_name) # hard labels for all mesh (2D list)
            seseg_name = ["{}_{}.seseg".format(d,i) for i in range(count)] # slabel
            seseg_dirs.append(seseg_name) # soft labels for all mesh (2D list)
            seg_name = [os.path.join(seg_path, elem) for elem in seg_path]
            face_seg = [np.loadtxt(elem, dtype=int) for elem in seg_name]
            # for elem in seg_res:
            #     fpath = os.path.join(seg_path, elem)
            #     part_label = np.loadtxt(fname=fpath, dtype=int)
            #     face_seg.append(part_label)
            face_labels.append(face_seg)

        return face_labels, eseg_dirs, seseg_dirs

    elif layer == 0: # dir_path would be path to seg_simp
        filenames = [d for d in os.listdir(dir_path)]
        sfilenames = sorted(filenames, key=lambda d: int(d.split(".")[0].split("_")[0]))
        seg_files = ["{}.npz".format(f.split(".")[0]) for f in sfilenames]
        eseg_dirs = []
        seseg_dirs = []
        face_labels = []
        for elem in seg_files:
            fpath = os.path.join(dir_path, elem)
            with np.load(fpath) as part_label:
                part_label_tag = part_label.files
                eseg_name = ["{}.eseg".format(f.split(".")[0]) for f in part_label_tag]
                seseg_name = ["{}.seseg".format(f.split(".")[0]) for f in part_label_tag]
                eseg_dirs.append(eseg_name)
                seseg_dirs.append(seseg_name)
                face_seg = [np.asarray(part_label[t]).reshape(-1) for t in part_label_tag]
                face_labels.append(face_seg)

        return face_labels, eseg_dirs, seseg_dirs
    

def save_mesh(dir_path, point, face, name):
    """_summary_: save mesh into dir_path with name

    Args:
        dir_path (str): saving directory
        point (ndarray): mesh point
        face (ndarray): mesh face
        name (str): name in str
    """
    name_path = os.path.join(dir_path, name)
    cur_mesh = trimesh.Trimesh(vertices=point, faces=face)
    cur_mesh.export(name_path)


def save_mult_labels(dir_path, labels, name, name_key):
    """_summary_: used to save multiple face label

    Args:
        dir_path (str): _description_
        labels (ndarray): _description_
        name (ndarray): _description_
        name_key (list): int list of name prefix
    """
    name_path = os.path.join(dir_path, name) # path for each name
    name_prefix = name.split("_")[0] # prefix for name (number)
    label_dict = {}
    for k, arr in zip(name_key, labels):
        k = str(k)
        nlabel = np.asarray(arr).reshape(-1).astype(np.int32)
        label_dict[k] = nlabel
    np.savez(name_path, **label_dict)

# def save_meshes(dir_path, points, faces, names):
#     for i in range(len(names)):
#         name_path = os.path.join(dir_path, names[i])
#         point = points[i]
#         face = faces[i]
#         cur_mesh = trimesh.Trimesh(vertices=point, faces=face)
#         cur_mesh.export(name_path)


# -----(getter)-----

def tri_to_o3d(tri_mesh):
    """_summary_: convert trimesh mesh into open3d mesh

    Args:
        mesh (trimesh mesh)
    """
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(tri_mesh.vertex_normals.copy())
    o3d_mesh.triangle_normals = o3d.utility.Vector3dVector(tri_mesh.face_normals.copy())
    
    return o3d_mesh


def get_vertex(mesh, library="trimesh"):
    """_summary_: get vertex from mesh (read through different library)

    Args:
        mesh (_type_): original mesh
        library (str, optional): imported library of mesh. Defaults to "trimesh".

    Returns:
        vert (ndarray): (n, 3) sized vertex
    """
    if library == "trimesh":
        return mesh.vertices
    elif library == "o3d":
        return np.asarray(mesh.vertices)
    

def get_face(mesh, library="trimesh"):
    """_summary_: get face from mesh (read through different library)

    Args:
        mesh (_type_): original mesh
        library (str, optional): imported library of mesh. Defaults to "trimesh".

    Returns:
        face (ndarray): (n, 3) sized face
    """
    if library == "trimesh":
        return mesh.faces
    elif library == "o3d":
        return np.asarray(mesh.triangles)
    

# def get_vnorm(mesh, library="trimesh"):
#     """_summary_: get vertex normals from mesh (read through different library)

#     Args:
#         mesh (_type_): original mesh
#         library (str, optional): imported library of mesh. Defaults to "trimesh".

#     Returns:
#         vertex normal (ndarray): (n, 3) sized vertex normal
#     """
#     if library == "trimesh":
#         return mesh.vertex_normals
#     elif library == "o3d":
#         return np.asarray(mesh.vertex_normals)
    

# def get_fnorm(mesh, library="trimesh"):
#     """_summary_: get face normals from mesh (read through different library)

#     Args:
#         mesh (_type_): original mesh
#         library (str, optional): imported library of mesh. Defaults to "trimesh".

#     Returns:
#         face normal (ndarray): (n, 3) sized face normal
#     """
#     if library == "trimesh":
#         return mesh.face_normals
#     elif library == "o3d":
#         return np.asarray(mesh.triangle_normals)
    

def read_eseg(dirpath, fname):
    """_summary_: reads single hard label

    Args:
        dirpath (str): directory path to be saved
        fname (str): file name

    Returns:
        eseg (ndarray): int label
    """
    fpath = os.path.join(dirpath, fname)
    eseg = np.loadtxt(fpath, dtype=np.int64)

    return eseg
    

def save_eseg(dirpath, fname, labels):
    """_summary_: saves single soft label

    Args:
        dirpath (str): directory path to be saved
        fname (str): _description_
        labels (ndarray): _description_
    """
    fpath = os.path.join(dirpath, fname)
    np.savetxt(fpath, labels, fmt="%d", newline="\n")


def read_seseg(dirpath, fname):
    fpath = os.path.join(dirpath, fname)
    seseg = np.loadtxt(fpath, dtype=np.float32)

    return seseg


def save_seseg(dirpath, fname, labels):
    fpath = os.path.join(dirpath, fname)
    num_classes = labels.shape[1]
    _format = " ".join(["%.6f"] * num_classes)
    np.savetxt(fpath, labels, fmt=_format, newline="\n")


def sort_mesh_name(meshes, names):
    """(Not Used) sorts name and meshes accordingly

    Args:
        meshes (python list): mesh in unsorted manner
        names (python list): file name in unsorted manner

    Returns:
        smeshes: sorted mesh
        smanes: sorted name (according to prefix integer number)
    """
    sort_key = np.char.partition(names, '_')[:, 0].astype(int)
    sort_idx = np.argsort(sort_key, kind='stable')
    smeshes = [meshes[i] for i in sort_idx]
    snames = [names[i] for i in sort_idx]

    return smeshes, snames


def get_fnames(dir_path):
    filenames = [d for d in os.listdir(dir_path)]
    sfilenames = sorted(filenames, key=lambda d: int(d.split(".")[0].split("_")[0]))

    return sfilenames

def create_new_label(cur_label, map_dict):
    len_label = cur_label.shape[0] # get shape
    new_label = np.empty(len_label, dtype=np.int64)
    # get mask
    for _k in map_dict.keys():
        mask = (cur_label == _k)
        new_label[mask] = map_dict[_k]

    return new_label

def get_label_number(lst_dict):
    # calculates segmentation number
    max_val = 0
    for _dict in lst_dict:
        _max = max(list(_dict.values()))
        if _max > max_val:
            max_val = _max
    seg_num = max_val+1

    return seg_num


# added for visualization
def read_mesh_alp(dir_path, only_pref=False, ext=(".off", ".ply", ".obj")):
    """_summary_: reads the mesh in alphabetical order

    Args:
        dir_path (str): directory path where meshes are
        supporting mesh file extensions: .obj, .stl, .ply, .off

    Returns:
        list of meshes (trimesh.Trimesh): list of meshes
        Trimesh
        ├── vertices [n, 3]
        ├── faces [m, 3]
        ├── vertex_normals [n, 3]
        ├── face_normals [m, 3]
        ├── edges / edges_unique
        ├── is_watertight
        ├── volume / area / centroid
        ├── methods: show(), export(), fill_holes(), split()
    """
    fnames = [f for f in os.listdir(dir_path) if f.endswith(ext)]
    # count = len(os.listdir(dir_path))
    # fnames = ["{}.off".format(f) for f in range(1, count+1)]
    # _key = [int(f.split(".")[0]) for f in fnames]
    # sfnames = sorted(fnames, key=lambda f: int(f.split(".")[0]))
    meshes = []
    for f in fnames:
        fpath = os.path.join(dir_path, f)
        print(fpath)
        mesh = trimesh.load(fpath)
        meshes.append(mesh)

    return meshes, fnames