import os
import argparse
import numpy as np
from util import pre_util
from util import edge_label as el
import visualize as visu
from tqdm import tqdm

## Visualize simplified mesh and corresponding segmentation result ##

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="prince_simp_1000") # "prince_ben", "prince_simp_1000"

args = parser.parse_args()

# Path
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "datasets", args.data_dir)
seg_path = os.path.join(data_path, "seg")
sseg_path = os.path.join(data_path, "sseg")

gt_path = ""
seg_res_path = ""
if args.data_dir == "prince_ben":
    gt_path = os.path.join(data_path, "gt")
    seg_res_path = os.path.join(data_path, "seg_res")
elif args.data_dir == "prince_simp_1000" or args.data_dir == "prince_simp_5000":
    gt_path = os.path.join(data_path, "gt_conn") # "gt_simp"
    seg_res_path = os.path.join(data_path, "seg_conn") # "seg_simp"

# make edge saving directory
os.makedirs(seg_path, exist_ok=True)
os.makedirs(sseg_path, exist_ok=True)

meshes, names = pre_util.read_mesh(gt_path) # read all meshes / sorted
if args.data_dir == "prince_ben":
    point_seg, eseg_name, seseg_name = pre_util.read_seg_res(seg_res_path, layer=1)
elif args.data_dir == "prince_simp_1000" or args.data_dir == "prince_simp_5000":
    point_seg, eseg_name, seseg_name = pre_util.read_seg_res(seg_res_path, layer=0)

for i in range(0,20): # len(meshes)
    # Map inconsistent label to specific semantic part for all 19 classes
    # 단위는 20씩 돌려주세요
    mesh = meshes[i]
    name = names[i]
    points = pre_util.get_vertex(mesh)
    faces = pre_util.get_face(mesh)
    cur_seg = point_seg[i] # get related segmentation
    # visu.vis_face_seg(points, faces, cur_seg[10])
    edges, etof, ftoe = el.build_edge_order(faces)
    # selected number of segmentation (from cur_seg)
    # Get num_lst and lst_dict from
    # simp_seg_label >> class#_###.txt
    # e.g. class 1
    num_lst = [3, 3, 3, 6, 3, 7, 11, 1, 4, 8, 8, 6, 2, 0, 0, 5, 5, 6, 2, 5]
    lst_dict = [
        {5:3, 0:0, 4:2, 3:2, 1:1, 2:1},
        {1:3, 0:0, 2:2, 3:2, 4:1, 5:1},
        {5:3, 0:0, 4:2, 3:2, 1:1, 2:1},
        {1:3, 0:0, 3:2, 2:2, 5:1, 4:1},
        {1:3, 0:0, 2:2, 3:2, 4:1, 5:1},
        {1:3, 0:0, 3:2, 4:2, 5:1, 6:1, 2:3},
        {1:3, 0:0, 3:2, 2:2, 4:1, 5:1},
        {5:3, 0:0, 4:2, 3:2, 2:1, 1:1},
        {5:3, 0:0, 3:2, 4:2, 2:1, 1:1},
        {3:3, 0:0, 2:2, 1:2, 5:1, 4:1},
        {5:3, 0:0, 4:2, 6:2, 2:1, 1:1, 3:1},
        {6:3, 0:0, 4:2, 5:2, 2:1, 1:1, 3:1},
        {5:3, 0:0, 4:2, 3:2, 1:1, 2:1},
        {5:3, 0:0, 3:2, 4:2, 1:1, 2:1},
        {3:3, 0:0, 2:2, 1:2, 5:1, 4:1},
        {5:3, 0:0, 1:2, 2:2, 4:1, 3:1},
        {3:3, 0:0, 1:2, 2:2, 5:1, 4:1},
        {1:3, 0:0, 3:2, 2:2, 4:1, 5:1},
        {5:3, 0:0, 3:2, 4:2, 1:1, 2:1},
        {5:3, 0:0, 3:2, 4:2, 1:1, 2:1},
    ]
    elem = num_lst[i%20] # elem = idx of cur_seg
    seg = cur_seg[elem]
    assert faces.shape[0] == cur_seg[elem].shape[0], "len vertices and len labels not same"
    cur_dict = lst_dict[i%20]
    new_seg = pre_util.create_new_label(seg, cur_dict)# convert seg using dictionary
    if i == 377: # merged ear label to fourleg obj of i=377
        seg2 = cur_seg[3]
        cur_dict2 = {7:2, 6:2, 10:100, 9:100, 8:100, 0:100, 3:100, 1:100, 14:100, 2:100, 4:100, 13:100, 12:100, 11:100, 5:100}
        new_seg2 = pre_util.create_new_label(seg2, cur_dict2)
        new_seg = np.minimum(new_seg, new_seg2)
    face_prob = el.get_face_probs(new_seg)

    seg_num = pre_util.get_label_number(lst_dict) # needed for padding seseg
    seseg = el.build_bound_slabel(etof, face_prob, edges, seg_num)
    # increase dimension to class label number
    eseg = seseg.argmax(axis=1)
    eseg_fname = eseg_name[i][elem]
    seseg_fname = seseg_name[i][elem]

    visu.vis_face_seg(points, faces, new_seg)
    print("Seg Name: {}.seg, Seg Num: {}".format(name.split(".")[0], len(set(seg))))

    pre_util.save_eseg(seg_path, "{}.eseg".format(name.split(".")[0]), eseg)
    pre_util.save_seseg(sseg_path, "{}.seseg".format(name.split(".")[0]), seseg)
        