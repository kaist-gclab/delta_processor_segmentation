import numpy as np
from datetime import datetime


def build_edge_order(faces):
    """_summary_

    Args:
        faces (ndarray): each face contains three vertex indices (num_vert, 3)
    Returns:
        edges (ndarray): vertex (u, v) in incresing order (num_edge, 2)
        etof (python double list): per each edge_id, contains relative face id
        ftoe (ndarray): for each face_id, contains tuple of three edge_id
    """
    visited = {}
    edges = []
    etof = [] # edge to face
    ftoe = [] # face to edge

    for fi, (a, b, c) in enumerate(faces):
        eids = []
        for u, v in ((a, b), (b, c), (c, a)):
            _key = (u, v) if u < v else (v, u)
            if _key not in visited:
                eid = len(edges) # 여태 누적된 edge 개수 = edge_id
                edges.append(_key)
                etof.append([fi]) # in current eid position, add list that contains face_id
                visited[_key] = eid
            else:
                eid = visited[_key] # get edge id
                etof[eid].append(fi)
            eids.append(eid)
        ftoe.append(tuple(eids)) # tuple of edge ids per face

    edges = np.array(edges, dtype=int)
    ftoe = np.array(ftoe, dtype=int)

    return edges, etof, ftoe


def build_hlabel(etof, flabels, mode="bound01", max_classes=None):
    """_summary_

    Args:
        etof (python list): for each edge, have list of face id assigned to it
        flabels (ndarray): face labels
        mode (str, optional): _description_. Defaults to "bound01".
            bound01 = 1 if dif label, 0 if one label (1=boundary edge)
            take_min
            pair_id
        max_classes (_type_, optional): _description_. Defaults to None.
    """
    num_edge = len(etof)
    out = np.zeros(num_edge, dtype=int)

    for e, flist in enumerate(etof):
        if len(flist) == 1: # only one face assigned to edge
            li = int(flabels[flist[0]]) # face_label of first-found face
            if mode == "bound01":
                out[e] = 1
            elif mode == "take_min":
                out[e] = li
            elif mode == "pair_id":
                out[e] = li if max_classes is None else li*(max_classes+1)+li
        else:
            li, lj = int(flabels[flist[0]]), int(flabels[flist[1]])
            if mode == "bound01":
                out[e] = int(li != lj)
            elif mode == "take_min":
                out[e] = min(li, lj)
            elif mode == "pair_id":
                a, b = (li, lj) if li <= lj else (lj, li)
                if max_classes is None:
                    out[e] = (a, b)
                else:
                    out[e] = a*(max_classes+1)+b

    return out


# helper function that makes one-hot vector for face label
def get_face_probs(flabels, smoothing=0.0):
    """_summary_

    Args:
        flabels (_type_): _description_
        smoothing (float, optional): _description_. Defaults to 0.0.
    Returns:
        face_prob (ndarray): one-hot vector of face probability (num_flabels, num_classes)
    """
    flabels = flabels.astype(int)
    num_classes = flabels.max()+1 # label 종류 개수
    num_lab = flabels.shape[0]

    if smoothing == 0.0:
        face_prob = np.zeros((num_lab, num_classes), dtype=np.float32)
        face_prob[np.arange(num_lab), flabels] = 1.0
        return face_prob
    else:
        eps = float(smoothing)
        face_prob = np.full((num_lab, num_classes), eps/(num_classes-1), dtype=np.float32)
        face_prob[np.arange(num_lab), flabels] = 1.0 - eps
        return face_prob


# Helper function for build_bound_slabel
def build_slabel(etof, face_prob, mode="mean"):
    num_edge = len(etof)
    num_classes = face_prob.shape[1]
    out = np.zeros((num_edge, num_classes), dtype=np.float32)

    for e, flist in enumerate(etof):
        part_fprob = face_prob[flist]
        if mode == "mean":
            out[e] = part_fprob.mean(axis=0)
        elif mode == "geom_mean":
            eps = 1e-8
            out[e] = np.exp(np.log(np.clip(part_fprob, eps, 1)).mean(axis=0))
            out[e] /= out[e].sum() + eps
    return out


# Creates seseg
def build_bound_slabel(etof, face_prob, edges, seg_len):
    b = build_slabel(etof, face_prob)

    b_ids = [e for e, fl in enumerate(etof) if len(fl) == 1]
    if not b_ids: # early stop
        shape_len = len(b.shape)
        if shape_len == 1:
            pw = seg_len-1
        else:
            pw = seg_len - b.shape[1]
        padded_array = np.pad(
            b, 
            pad_width=((0, 0), (0, pw)), 
            mode='constant', 
            constant_values=0
        )
        return padded_array #b
    
    # I have to read code later
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, c):
        ra, rc = find(a), find(c)
        if ra != rc: parent[rc] = ra

    # vertex -> boundary edges touching it
    v2e = {}
    for e in b_ids:
        u, v = edges[e]
        v2e.setdefault(u, []).append(e)
        v2e.setdefault(v, []).append(e)
    # union all edges per vertex bucket
    for lst in v2e.values():
        for i in range(1, len(lst)):
            union(lst[0], lst[i])

    # average within each component
    comps = {}
    for e in b_ids:
        comps.setdefault(find(e), []).append(e)
    for comp in comps.values():
        m = float(b[comp].mean())
        b[comp] = m

    shape_len = len(b.shape)
    if shape_len == 1:
        pw = seg_len-1
    else:
        pw = seg_len - b.shape[1]
    padded_array = np.pad(
        b, 
        pad_width=((0, 0), (0, pw)), 
        mode='constant', 
        constant_values=0
    )
    
    return padded_array # b


def edge_multiplicity(faces):
    """_summary_
    Args:
        faces (ndarray): disconnected / connected faces in mesh
    Returns:
        E_u: unique edge array
        inv: unique inverse
        counts (int): number of unique edge counts
    """
    faces = np.asarray(faces, dtype=int).reshape(-1,3)
    e01 = np.sort(faces[:, [0,1]], axis=1)
    e12 = np.sort(faces[:, [1,2]], axis=1)
    e20 = np.sort(faces[:, [2,0]], axis=1)
    E = np.vstack([e01, e12, e20])
    uniq_edge, uniq_inv, counts = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    return uniq_edge, counts, uniq_inv.reshape(-1,3)


def weld_vertices_with_labels(vert, face, face_labels=None, eps=1e-8, drop_duplicates=True):
    """Remove duplicates and restore topology (in some cases faces are disconnected so connect them)

    vert: (num_vert, 3) float array of vertices
    face: (num_face, 3) int array of triangle vertex indices
    face_labels: (num_face,) optional int labels per face
    eps: welding tolerance (merge verts whose coordinates differ < eps)
    drop_duplicates: if True, remove duplicate faces after welding

    Returns:
        vert_w: (Nw,3) welded vertices
        face_w: (Mw,3) remapped faces (degenerate and optional duplicate faces removed)
        labels_w: (Mw,) face labels filtered to match face_w (or None if face_labels is None)
        kept_face_idx: indices into original faces that were kept (labels_w = face_labels[kept_face_idx])
        old_to_new_vidx: (num_vert,) array mapping old vertex id -> new (welded) vertex id
    """
    vert = np.asarray(vert, float).reshape(-1, 3)
    face = np.asarray(face, int).reshape(-1, 3)
    num_face = face.shape[0]

    key = np.round(vert / eps).astype(np.int64) # quantize vertex into eps (diff less than eps would be considered as same vertex)
    uniq_key, vert_idx, old_to_new_vidx = np.unique(key, axis=0, return_index=True, return_inverse=True) # find unique vertex index and its mapping
    vert_w = vert[vert_idx] # vertex weld
    face_w = old_to_new_vidx[face] # map face -> vertex weld id

    # vertex weld로 바꾼 후 edge의 시작과 끝이 같은 경우 - degenerate (좋지 못한 퀄리티의 face)
    non_deg = (face_w[:, 0] != face_w[:, 1]) & (face_w[:, 1] != face_w[:, 2]) & (face_w[:, 2] != face_w[:, 0]) # ndarray (bool)
    face_w = face_w[non_deg] # degenerate face 제거
    kept_face_idx = np.nonzero(non_deg)[0] # non-degenerate face index

    # remove duplicate faces (same 3 ids independent of order)
    if drop_duplicates and len(face_w) > 0:
        face_sorted = np.sort(face_w, axis=1) # sort vertex coordinate within each face
        _, uniq_idx = np.unique(face_sorted, axis=0, return_index=True) # index of unique face
        uniq_idx = np.sort(uniq_idx) # 현재 uniq_idx는 face안의 vertex 작은 순으로 되어버림
        face_w = face_w[uniq_idx] # 원래 face order대로 indexing
        kept_face_idx = kept_face_idx[uniq_idx] # 살아남은 face index (to use on labeling)

    # re-calculate face labels
    if face_labels is None:
        labels_w = None
    else:
        face_labels = np.asarray(face_labels)
        labels_w = face_labels[kept_face_idx]

    return vert_w, face_w, labels_w, kept_face_idx, old_to_new_vidx


def select_idx(len_label, percent=0.03):
    seed = datetime.now().year # get year
    np.random.seed(seed) # set seed
    noise_size = int(np.ceil(len_label * percent)) # number of noise
    lidx = np.random.choice(len_label, noise_size, replace=False)

    return lidx


def create_dict(class_num):
    cdict = dict()
    for i in range(class_num):
        if i == class_num-1:
            cdict[i] = 0
        else:
            cdict[i] = i+1

    return cdict


def noise_seg(seg, sseg, idx, class_num):
    seed = datetime.now().year # get year
    np.random.seed(seed) # set seed

    cdict = create_dict(class_num)
    new_labels = np.array([cdict[seg[i]] for i in idx]) # new label
    seg[idx] = new_labels
    # sseg[idx[:, None], new_labels] = 1.0

    rows = np.asarray(idx, dtype=np.int64).ravel()
    cols = np.asarray(new_labels, dtype=np.int64).ravel()
    sseg[rows, cols] = 1.0
    # new_soft_label = np.zeros((len(seg), class_num), dtype=np.float32)
    # new_soft_label[rows, cols] = 1.0
    # for elem in idx: # 이전 레이블과 비교
    #     print("Before: {}, After: {}".format(sseg[elem], new_soft_label[elem]))
    
    return seg, sseg


# Convert edge label back to face label
def build_flabel_from_edges(etof, edge_labels, mode="majority", fill_value=-1):
    """
    Build face labels from per-edge hard labels.

    Args:
        etof (list of list[int]): for each edge, list of incident face ids.
        edge_labels (array-like): integer label per edge, shape (num_edges,).
        mode (str): "majority", "min", or "max" to aggregate edge labels.
        fill_value (int): label used for faces that have no incident edges.

    Returns:
        flabels (ndarray[int]): label per face, shape (num_faces,).
    """
    edge_labels = np.asarray(edge_labels)
    # infer number of faces from etof
    max_fid = -1
    for flist in etof:
        if flist:
            max_fid = max(max_fid, max(flist))
    num_faces = max_fid + 1

    # build face -> incident edges mapping
    ftoe = [[] for _ in range(num_faces)]
    for e, flist in enumerate(etof):
        for f in flist:
            ftoe[f].append(e)

    flabels = np.full(num_faces, fill_value, dtype=int)

    for f, edges in enumerate(ftoe):
        if not edges:
            continue  # isolated face, stays fill_value
        labs = edge_labels[edges]

        if mode == "majority":
            vals, counts = np.unique(labs, return_counts=True)
            flabels[f] = vals[np.argmax(counts)]
        elif mode == "min":
            flabels[f] = int(labs.min())
        elif mode == "max":
            flabels[f] = int(labs.max())
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

    return flabels