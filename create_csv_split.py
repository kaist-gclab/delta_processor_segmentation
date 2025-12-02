import argparse, csv, os, random
from datetime import datetime
from pathlib import Path
import pandas as pd


def get_obj_fname(obj_path, ext=".obj"):
    fnames = [f for f in os.listdir(obj_path) if f.endswith('.obj')]
    sfnames = sorted(fnames, key=lambda f: int(f.split(".")[0].split("_")[0]))

    return sfnames


def get_split(data_path, src_path, test_ratio, seed=2025, class_size=20):
    """_summary_

    Args:
        data_path (python path): path to the current dataset
        src_path (python path): path to .obj file directory
        test_ratio (float): ratio of the test file
        seed (int): Seed for random. Defaults to 2025.
        class_size (int): class size in sorted datset. Defaults to 20
    """
    rng = random.Random(seed)
    split_dict = {"train": [], "test": []}

    sfnames = get_obj_fname(src_path) # get all file names
    test_num = round(class_size*test_ratio)
    # train_num = class_size-test_num

    for i in range(0, len(sfnames), class_size):
        cur_class = sfnames[i:i+class_size]
        rng.shuffle(cur_class)

        test_items = cur_class[:test_num]
        stest_items = sorted(test_items, key=lambda f: int(f.split(".")[0].split("_")[0]))
        train_items = cur_class[test_num:]
        strain_items = sorted(train_items, key=lambda f: int(f.split(".")[0].split("_")[0]))

        def to_dict(fname):
            return {"model": str(fname)}

        split_dict["test"].extend(map(to_dict, stest_items))
        split_dict["train"].extend(map(to_dict, strain_items))

    return split_dict


def write_csv(split_dict, csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "model"])
        w.writeheader()
        for split, rows in split_dict.items():
            for r in rows:
                row = {"split": split, **r}
                w.writerow(row)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="prince_simp_1000")
    parser.add_argument("--test_ratio", type=int, default=0.2)
    parser.add_argument("--csv_name", type=str, default="train_test_split.csv")

    args = parser.parse_args()

    now = datetime.now()
    seed = now.year

    # Path
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "datasets", args.data_dir)
    csv_path = os.path.join(data_path, args.csv_name)
    if args.data_dir == "prince_simp_1000":
        gt_path = os.path.join(data_path, "gt_simp")


    if not os.path.exists(csv_path): # check if .csv file exist and generate one if not
        csv_data = get_split(data_path, gt_path, args.test_ratio, seed)
        write_csv(csv_data, csv_path)

    # # load csv file
    # df = pd.read_csv(csv_path)

    # train_objs = df.loc[df["split"]=="train", "model"].tolist()
    # test_objs  = df.loc[df["split"]=="test",  "model"].tolist()