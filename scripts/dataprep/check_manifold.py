import os
import utils_dp as ut
import mesh_fun as mf

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "data")
simp_path = os.path.join(data_path, "gt_simp")

# mfiles = os.listdir(simp_path)
mfiles = [f for f in os.listdir(simp_path) if f.lower().endswith(".obj")]
mfiles.sort()


# logging file
log_path = os.path.join(data_path, "manifold.txt")
f = open(log_path, 'w')
for mesh_file in mfiles:
    # create full path
    mpath = os.path.join(simp_path, mesh_file)
    
    # read off file
    vert, face = ut.read_obj(mpath)
    
    water, norm_con = mf.check_manifold(vert, face)
    f.write("File {}: Watertight {}, norm_dir {}\n".format(mesh_file[:-4], water, norm_con))
f.close()
