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
parser.add_argument("--data_dir", type=str, default="pclass") # "prince_ben", "prince_simp_1000"

args = parser.parse_args()

# Path
base_path = os.path.dirname(__file__)

for i in range(15, 16):
    dir_name = "{}{}".format(args.data_dir, i)
    cdata_path = os.path.join(base_path, "datasets", dir_name)
    cseg_path = os.path.join(cdata_path, "seg")
    csseg_path = os.path.join(cdata_path, "sseg")

    ndir_name = "noise_{}{}".format(args.data_dir, i)
    ndata_path = os.path.join(base_path, "datasets", ndir_name)
    nseg_path = os.path.join(ndata_path, "seg")
    nsseg_path = os.path.join(ndata_path, "sseg")

    # make noise edge label saving directory
    os.makedirs(nseg_path, exist_ok=True)
    os.makedirs(nsseg_path, exist_ok=True)

    # load previous seg, sseg files
    obj_filenmaes = [f for f in os.listdir()]
    seg_filenames = [f for f in os.listdir(cseg_path)]
    sseg_filenames = [f for f in os.listdir(csseg_path)]
    sorted_seg = sorted(seg_filenames, key=lambda f: int(f.split(".")[0].split("_")[0]))
    sorted_sseg = sorted(sseg_filenames, key=lambda f: int(f.split(".")[0].split("_")[0]))
    for j in range(20):
        seg_name = sorted_seg[j]
        sseg_name = sorted_sseg[j]
        ceseg = pre_util.read_eseg(cseg_path, seg_name)
        cseseg = pre_util.read_seseg(csseg_path, sseg_name)
        
        # class num
        len_label, class_num = cseseg.shape
        # select 3% of random index
        idx = el.select_idx(len_label)
        neseg, nseseg = el.noise_seg(ceseg, cseseg, idx, class_num)
        # visu.vis_face_seg(points, faces, new_seg)

        pre_util.save_eseg(nseg_path, seg_name, neseg)
        pre_util.save_seseg(nsseg_path, sseg_name, nseseg)

        # print("Seg Name: {}.seg, Seg Num: {}".format(seg_name.split(".")[0]))
    
        