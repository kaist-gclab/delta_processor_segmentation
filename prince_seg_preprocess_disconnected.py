import os
import argparse
from tqdm import tqdm
from util import pre_util
from util import edge_label as el

## Connects disconnected face for simplified mesh ##
## 기하적으로 연결되어있으니 각 face가 독립적으로 생성되는 문제를 고치기 위한 코드 ##

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="prince_simp_1000") # "prince_ben", "prince_simp_1000"

args = parser.parse_args()

# Path
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "datasets", args.data_dir) # dataset dir
nmesh_path = os.path.join(data_path, "gt_conn") # connected obj mesh
nseg_path = os.path.join(data_path, "seg_conn") # connected label

gt_path = ""
seg_res_path = ""
if args.data_dir == "prince_ben":
    gt_path = os.path.join(data_path, "gt")
    seg_res_path = os.path.join(data_path, "seg_res")
else:
    gt_path = os.path.join(data_path, "gt_simp")
    seg_res_path = os.path.join(data_path, "seg_simp")

# make edge saving directory
os.makedirs(nmesh_path, exist_ok=True)
os.makedirs(nseg_path, exist_ok=True)

meshes, names = pre_util.read_mesh(gt_path, only_pref=True) # read all meshes
if args.data_dir == "prince_ben":
    point_seg, dirnames = pre_util.read_seg_res(seg_res_path, layer=1)
else:
    point_seg, segnames, seglabels = pre_util.read_seg_res(seg_res_path, layer=0)


for i in tqdm(range(len(meshes))):
    cur_mesh = meshes[i]
    cur_vert = pre_util.get_vertex(cur_mesh)
    cur_face = pre_util.get_face(cur_mesh)
    cur_seg = point_seg[i] # list of segmentations
    
    new_vert, new_face, _, label_index, _ = el.weld_vertices_with_labels(cur_vert, cur_face)
    new_labels = []
    for j in range(len(point_seg[i])):
        new_label = point_seg[i][j][label_index]
        new_labels.append(new_label)
    pre_util.save_mesh(nmesh_path, new_vert, new_face, names[i])
    pre_util.save_mult_labels(nseg_path, new_labels, segnames[i], seglabels[i])