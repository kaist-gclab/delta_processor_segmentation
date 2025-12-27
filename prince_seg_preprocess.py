import os
import argparse
from tqdm import tqdm
from util import pre_util
from util import edge_label as el

## Created edge segmentation and soft edge segmentation ##

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="prince_simp_1000") # "prince_ben", "prince_simp_1000"

args = parser.parse_args()

# Path
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "datasets", args.data_dir) # dataset dir
seg_path = os.path.join(data_path, "seg") # hard edge label dir
sseg_path = os.path.join(data_path, "sseg") # soft edge label dir

gt_path = ""
seg_res_path = ""
if args.data_dir == "prince_ben":
    gt_path = os.path.join(data_path, "gt")
    seg_res_path = os.path.join(data_path, "seg_res")
else:
    gt_path = os.path.join(data_path, "gt_simp")
    seg_res_path = os.path.join(data_path, "seg_simp")

# make edge saving directory
os.makedirs(seg_path, exist_ok=True)
os.makedirs(sseg_path, exist_ok=True)

meshes, names = pre_util.read_mesh(gt_path) # read all meshes
# file structure differ (layer)
if args.data_dir == "prince_ben":
    point_seg, eseg_name, seseg_name = pre_util.read_seg_res_with_eseg_fname(seg_res_path, layer=1)
else:
    point_seg, eseg_name, seseg_name = pre_util.read_seg_res_with_eseg_fname(seg_res_path, layer=0)

for i in tqdm(range(len(meshes))):
    cur_mesh = meshes[i]
    cur_face = pre_util.get_face(cur_mesh)
    cur_seg = point_seg[i] # list of segmentations
    edges, etof, ftoe = el.build_edge_order(cur_face)
    for j in range(len(cur_seg)):
        # check if face number and label numbers are same
        assert cur_face.shape[0] == cur_seg[j].shape[0], "len vertices and len labels not same"
        # calculate face prob to get soft edge label
        face_prob = el.get_face_probs(cur_seg[j])
        seseg = el.build_bound_slabel(etof, face_prob, edges) # soft edge prob
        eseg = seseg.argmax(axis=1) # hard label
        eseg_fname = eseg_name[i][j] # create soft fname
        seseg_fname = seseg_name[i][j] # create hard fname
        pre_util.save_eseg(seg_path, eseg_fname, eseg) # save soft edge label
        pre_util.save_seseg(sseg_path, seseg_fname, seseg) # save hard edge label
