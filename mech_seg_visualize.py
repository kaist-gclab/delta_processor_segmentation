
import os
import sys
from options.test_options import TestOptions
import util.pre_util as pre_util
import util.edge_label as el
import visualize as visu

from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer



def build_opt(dataroot, name, arch="meshunet", dataset_mode="segmentation", ncf=(32, 64, 128, 256),
    ninput_edges=1500, pool_res=(1050, 600, 300), resblocks=3, batch_size=4):
    test_argv = [
        "mech_seg_visualize.py",
        "--dataroot", dataset_path,
        "--name", name,
        "--arch", "meshunet",
        "--dataset_mode", "segmentation",
        "--ncf", "32", "64", "128", "256",
        "--ninput_edges", "1500",
        "--pool_res", "1050", "600", "300",
        "--resblocks", "3",
        "--batch_size", "4",
    ]

    old_argv = sys.argv
    sys.argv = test_argv
    try:
        opt = TestOptions().parse()
    finally:
        sys.argv = old_argv

    opt.serial_batches = True
    
    return opt


def run_test_with_opt(opt=None, epoch=-1):
    print('Running Test')
    # If no opt is given, fall back to command-line parsing (for normal CLI use)
    if opt is None:
        opt = TestOptions().parse()

    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)

    writer.reset_counter()
    pred_labels = []
    gt_labels = []

    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples, pred_label, gt_label = model.test()
        pred_labels.append(pred_label)
        gt_labels.append(gt_label)
        writer.update_counter(ncorrect, nexamples)

    f1 = model.calculate_f1(pred_labels, gt_labels)
    writer.print_acc(epoch, writer.acc)
    writer.print_f1(epoch, f1)

    return pred_label, writer.acc, f1


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "datasets")
    check_path = os.path.join("pclass_best")

    for i in range(1, 20):
        class_dir = "pclass{}".format(i)
        dataset_path = os.path.join(data_path, class_dir)
        cur_check_path = os.path.join(check_path, class_dir)
        opt = build_opt(dataset_path, cur_check_path)
        label, acc, f1 = run_test_with_opt(opt=opt, epoch=-1)
        nlabel = label.cpu().detach().numpy() # detach and convert to ndarray
        # convert edge label into point label

        test_mesh_path = os.path.join(dataset_path, "test")
        meshes, names = pre_util.read_mesh_alp(test_mesh_path)
        for j in range(len(meshes)):
            mesh = meshes[j]
            edge_label = nlabel[j]
            points = pre_util.get_vertex(mesh)
            faces = pre_util.get_face(mesh)
            edges, etof, ftoe = el.build_edge_order(faces)
            seg = el.build_flabel_from_edges(etof, edge_label)
            visu.vis_face_seg(points, faces, seg)