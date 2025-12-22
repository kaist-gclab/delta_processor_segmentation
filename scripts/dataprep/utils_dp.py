import os
import numpy as np
import trimesh

def read_off(fname):
    f = open(fname, 'r')

    lines = f.readlines()

    if lines[0].strip().startswith("OFF"):
        lines[0] = lines[0][3:]
    else:
        raise ValueError("Not a OFF file")
    
    nv, nf, ne = map(int, lines[1].rstrip().split())

    # off file has vertex coord info for next nv lines
    # and face information for next nf lines
    # numvert [vertidx in range nvert]

    vertices = []
    for i in range(2, nv+2):
        if not lines[i].strip().startswith("#"):
            vx, vy, vz = map(float, lines[i].rstrip().split())
            vertices.append([vx, vy, vz])
    vert = np.array(vertices)

    faces = []
    for j in range(nv+2, nv+nf+2):
        if not lines[j].strip().startswith("#"):
            n, vidx1, vidx2, vidx3 = map(int, lines[j].rstrip().split())
            faces.append([vidx1, vidx2, vidx3])
    face = np.array(faces)

    f.close()

    return vert, face


def read_obj(fname):
    mesh = trimesh.load(fname)

    return mesh.vertices, mesh.faces


def read_seg(sdir_path):
    sfiles = os.listdir(sdir_path)
    str_name = os.path.basename(sdir_path)
    print(str_name)
    all_seg = []
    for j in range(len(sfiles)):
        sfile = "{}_{}.seg".format(str_name, j)
        # create full path
        spath = os.path.join(sdir_path, sfile)
        # read and append to current array
        seg = np.loadtxt(spath, dtype=int)
        all_seg.append(seg)

    all_seg = np.concatenate(all_seg)

    return all_seg


def save_obj(fname, verts, faces):
    f = open(fname, 'w')
    for vert in verts:
        f.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))

    for face in faces:
        # object face has 1-based indexing
        face_idx = face + 1
        f.write("f {}\n".format(" ".join(map(str, face_idx))))