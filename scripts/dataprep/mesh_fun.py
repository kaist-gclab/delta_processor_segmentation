import numpy as np
import trimesh
from collections import defaultdict


def check_manifold(vert, face):
    mesh = trimesh.Trimesh(vertices=vert, faces=face)
    water = mesh.is_watertight
    norm_con = mesh.is_winding_consistent
    
    return water, norm_con


def mface_normal(vert, face):
    # calculate face normal (normalized) and face area
    a = vert[face[:, 0]]
    b = vert[face[:, 1]]
    c = vert[face[:, 2]]

    ba = b - a
    ca = c - a

    normal = np.cross(ba, ca)
    norm_normal = np.linalg.norm(normal)
    n_normalize = normal / norm_normal

    area = norm_normal / 2

    return n_normalize, area


def vec_angles(v1, v2):
    # calculate angle between two vector
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    dot_prod = np.einsum('ij, ij->i', v1, v2)
    cos_theta = dot_prod / (norm_v1*norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rad_angle = np.arccos(cos_theta)

    return rad_angle


def distance_point_plane(point, plane_point, plane_normal):
    # calculate distance between point and planes
    # Assumes point is (N, 3), plane_point and normal are (3,)
    diff = point - plane_point
    return np.dot(diff, plane_normal)


def project_point_on_plane(points, plane_point, plane_normal):
    # project points on plane
    # Supports broadcasting: points = (N, 3), plane_point = (3,), plane_normal = (3,)
    to_plane = points - plane_point
    distance = np.dot(to_plane, plane_normal)
    return points - np.outer(distance, plane_normal)


def project_to_2d(points, ignore_axis=2):
    # allv2 / plp2
    return np.delete(points, ignore_axis, axis=1)


def distance_point_edge3d(points, edge_start, edge_end):
    # calculate distance between point and edge (3d)
    # points: (N, 3), edge_start/end: (3,)
    edge_vec = edge_end - edge_start
    edge_len_sq = np.dot(edge_vec, edge_vec)
    vec_to_points = points - edge_start
    t = np.clip(np.dot(vec_to_points, edge_vec) / edge_len_sq, 0, 1)
    projections = edge_start + np.outer(t, edge_vec)
    return np.linalg.norm(points - projections, axis=1)


def generate_edges_with_segmentation(faces, face_seg):
    """
    Generate unique undirected edges from faces and assign segmentation
    label to each edge by voting from adjacent faces.
    
    Parameters:
    - faces: (F, 3) triangle face indices
    - face_seg: (F,) segmentation label for each face
    
    Returns:
    - unique_edges: (E, 2) array of unique edges
    - edge_labels: (E,) array of labels for each edge
    """
    # Step 1: List all edges and track their parent face
    edge_to_faces = []
    all_edges = []
    
    for i, face in enumerate(faces):
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            v1, v2 = sorted([face[a], face[b]])
            all_edges.append((v1, v2))
            edge_to_faces.append(i)
    
    all_edges = np.array(all_edges)
    edge_to_faces = np.array(edge_to_faces)

    # Step 2: Find unique edges and inverse map
    unique_edges, inverse_indices = np.unique(all_edges, axis=0, return_inverse=True)
    
    # Step 3: Aggregate segmentation labels per unique edge
    num_edges = unique_edges.shape[0]
    edge_labels = np.zeros(num_edges, dtype=int)

    for idx in range(num_edges):
        face_ids = edge_to_faces[inverse_indices == idx]
        labels = face_seg[face_ids]
        edge_labels[idx] = np.bincount(labels).argmax()  # majority vote

    return unique_edges, edge_labels


# functions from mesh_prepare.py
# ------------------------------
def get_edge_faces(faces):
    # create [v1, v2, face_a, face_b] np array
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