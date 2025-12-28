import numpy as np
import os
import ntpath
from tqdm import tqdm


def fill_mesh(mesh2fill, file: str, opt):
    load_path = get_mesh_path(file, opt.num_aug)
    if os.path.exists(load_path):
        mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True)
    else:
        mesh_data = from_scratch(file, opt)
        # convert to ndarray (for gemm_edges and ve)
        mesh_data.gemm_edges = np.array(mesh_data.gemm_edges, dtype=np.int64)
        mesh_data.ve = [np.array(ve_i, dtype=np.int32) for ve_i in mesh_data.ve]
        mesh_data.ve = np.array(mesh_data.ve, dtype=object)
        # print(mesh_data.ve)
        try:
            tmp_path = "".join(["tmp_", load_path])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            np.savez_compressed(tmp_path, gemm_edges=mesh_data.gemm_edges, vs=mesh_data.vs, edges=mesh_data.edges,
                                edges_count=mesh_data.edges_count, ve=mesh_data.ve, v_mask=mesh_data.v_mask,
                                filename=mesh_data.filename, sides=mesh_data.sides,
                                edge_lengths=mesh_data.edge_lengths, edge_areas=mesh_data.edge_areas,
                                features=mesh_data.features)
            os.rename(tmp_path, load_path)
        except:
            tqdm.write("failed")
    mesh2fill.vs = mesh_data['vs']
    mesh2fill.edges = mesh_data['edges']
    mesh2fill.gemm_edges = mesh_data['gemm_edges']
    mesh2fill.gemm_edges = mesh2fill.gemm_edges.astype(np.int32)
    mesh2fill.edges_count = int(mesh_data['edges_count'])
    mesh2fill.ve = mesh_data['ve']
    mesh2fill.ve = [np.array(ve_i, dtype=np.int32) for ve_i in mesh2fill.ve]
    mesh2fill.v_mask = mesh_data['v_mask']
    mesh2fill.filename = str(mesh_data['filename'])
    mesh2fill.edge_lengths = mesh_data['edge_lengths']
    mesh2fill.edge_areas = mesh_data['edge_areas']
    mesh2fill.features = mesh_data['features']
    mesh2fill.sides = mesh_data['sides']

def get_mesh_path(file: str, num_aug: int):
    filename, _ = os.path.splitext(file)
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    load_dir = os.path.join(dir_name, 'cache')
    load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, num_aug)))
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    return load_file

def from_scratch(file, opt):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.vs, faces = fill_from_file(mesh_data, file)
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    if opt.num_aug > 1:
        faces = augmentation(mesh_data, opt, faces)
    build_gemm(mesh_data, faces, face_areas)
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
    mesh_data.features = extract_features(mesh_data)
    return mesh_data

def fill_from_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def remove_non_manifolds(mesh, faces):
    """_summary_

    Args:
        mesh (_type_): _description_
        faces (_type_): _description_

    Returns:
        _type_: _description_
    """
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(mesh, faces)
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask]


def build_gemm(mesh, faces, face_areas):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas) #todo whats the difference between edge_areas and edge_lenghts?


def compute_face_normals_and_areas(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_normals /= face_areas[:, np.newaxis]
    assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_normals, face_areas


# Data augmentation methods
def augmentation(mesh, opt, faces=None):
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        faces = flip_edges(mesh, opt.flip_edges, faces)
    return faces


def post_augmentation(mesh, opt):
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)


def slide_verts(mesh, prct):
    """_summary_: vertex sliding
    randomly moves bit along to the incident edges

    Args:
        mesh (_type_): _description_
        prct (_type_): _description_
    """
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(mesh, edge_points).squeeze() #todo make fixed_division epsilon=0
    # thr = np.mean(dihedral) + np.std(dihedral) # threshold
    vids = np.random.permutation(len(mesh.ve)) # process in randomized order
    target = int(prct * len(vids)) # how many vertex to move
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]
            if min(dihedral[edges]) > 2.65: # fixed threshold aroung 152 deg
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                # slide vertex towards target
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) * (mesh.vs[vi_t] - mesh.vs[vi]) # new vertex
                mesh.vs[vi] = nv # assign
                shifted += 1 # count shifted vertices
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)


def scale_verts(mesh, mean=1, var=0.1):
    """_summary_: data augmentation helper method"""
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)


def angles_from_faces(mesh, edge_faces, faces):
    """_summary_: computes dihedral angle"""
    normals = [None, None]
    for i in range(2):
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    """_summary_: data augmentation method
    randomly flips the edge, without influencing edge connectivity
    
    Args:
        mesh (obj): object with vertices
        prce (fraction): percent
        faces (ndarray): (num_face, 3)
    """
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count) # randomly choose edge to flip
    # print(dihedral.min())
    # print(dihedral.max())
    target = int(prct * edge_count) # target number of flip
    flipped = 0 # coutner
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7: # threshold above 2.7
            edge_info = edge_faces[edge_key] # (edge_num, 4) - v0, v1, fA, fB
            if edge_info[3] == -1: # only one adj face, no flip
                continue
            # compute other two vertices
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]])))) # faces sharing the edge
            if new_edge in edges_dict: # if new diag already exists, duplicate connectivity -> don't flip
                continue
            # two triangles that would replace original
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])] # delete original edge
                edge_info[:2] = [new_edge[0], new_edge[1]] # convert edge info to new edges
                edges_dict[new_edge] = edge_key # add new edge to dictionary
                # two neighboring faces (change into inverted edge vertex)
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                # inverted edge가 포함된 모든 face에 대하여
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id] # get the face
                    for j in range(3): # update edge neighbor information
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3]))) # tuple of edge in sorted order
                        if cur_edge != new_edge: # updated edge
                            cur_edge_key = edges_dict[cur_edge] # lookup entry
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]: # pick up according faces
                                    edge_faces[cur_edge_key, 2 + idx] = face_id # update adjacency of face
                flipped += 1
    # print(flipped)
    return faces


# ------Helper function for flip_edges------
def rebuild_face(face, new_face):
    """_summary_: re-write new_face (edge flipped)
    replace one vertex that changed during flip
    face and new face differ by one vertex
    edge를 뒤집으면서 바뀐 vertex index를 이웃한 face에도 적용

    Args:
        face (ndarray): (3,) old face comp
        new_face (ndarray): (3,) new face component

    Returns:
        face: face with flipped edge vertex inserted
    """
    new_point = list(set(new_face) - set(face))[0] # finds face that is in new_face but not on old face
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face

def check_area(mesh, faces):
    """_summary_: checks if triangle is not too skinny, degenerate, inverted

    Args:
        mesh (object that consists ndarray): uses vertex
        faces (ndarray): (num_f, 3)

    Returns:
        bool: if area is okay
    """
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    """_summary_

    Args:
        faces (ndarray): (num_f, 3)

    Returns:
        edge_count (int): number of edges
        edge_faces (ndarray): per edge, write two adjacen face (num_e, 4)
        edge2keys (dict): have edge as key, specific edge count as value
    """
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys
# -----------------------------------------------------


# Helper function of extract features
def set_edge_lengths(mesh, edge_points=None):
    """_summary_: calculates all edge length per each edge and save into mesh.edge_length info

    Args:
        mesh (_type_): _description_
        edge_points (_type_, optional): _description_. Defaults to None.
    Returns:
        None: calculates edge length and write i mesh.edge lengths
    """
    if edge_points is not None:
        edge_points = get_edge_points(mesh) # edge_points: (num_e, 2)
    edge_lengths = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    mesh.edge_lengths = edge_lengths # (num_e, ) each entry is Eucledian dist of edge


def extract_features(mesh):
    """_summary_

    Args:
        mesh (obj): mesh obj with vertices

    Raises:
        ValueError: any error during extract and adding feature

    Returns:
        None: update feature information
    """
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def dihedral_angle(mesh, edge_points):
    """_summary_

    Args:
        mesh (obj): mesh object. vertices can be retrieved
        edge_points (nearray): ndarray of index of two vertex

    Returns:
        angles (ndarray): list of dihedral angles
    """
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def get_edge_points(mesh):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id 
        each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points


def get_side_points(mesh, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals

def get_opposite_angles(mesh, edge_points, side):
    """_summary_

    Args:
        mesh (obj): _description_
        edge_points (ndarray): (num_e, 2)
        side (_type_): _description_

    Returns:
        _type_: _description_
    """
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                   ord=2, axis=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths

def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
