import os
import utils_dp as ut

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "data")

# prepare file names
off_path = os.path.join(data_path, "gt")
obj_path = os.path.join(data_path, "gt_obj")
# create directory for obj path
os.makedirs(obj_path, exist_ok=True)

mfiles = os.listdir(off_path) # off files
# sfolds = os.listdir(fseg_path)
# sort just in case (their numbers are same / extension is different)
mfiles.sort()
# sfolds.sort()

for mesh_file in mfiles:
    # create full path
    mpath = os.path.join(off_path, mesh_file)
    
    # read off file
    vert, face = ut.read_off(mpath)
    
    # create save path
    npath = os.path.join(obj_path, "".join([mesh_file[:-4], ".obj"]))

    # save into obj file
    ut.save_obj(npath, vert, face)
