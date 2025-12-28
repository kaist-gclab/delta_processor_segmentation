from __future__ import print_function
import torch
import numpy as np
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

MESH_EXTENSIONS = [
    '.obj',
]

def is_mesh_file(filename):
    """_summary_: checks if the filename extension is mesh file

    Args:
        filename (str): names of the files

    Returns:
        bool: False if any ext is not mesh extension
    """
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

def pad(input_arr, target_length, val=0, dim=1):
    """_summary_

    Args:
        input_arr (ndarray): shape can be any (3D usually)
        target_length (_type_): _description_
        val (int, optional): padding value. Defaults to 0.
        dim (int, optional): padding axis. Defaults to 1.

    Returns:
        padded array(ndarray): padded array with target_len-cur_len on dim=dim, filling val = val
    """
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)

def seg_accuracy(predicted, ssegs, meshes):
    """_summary_: weight edge by edge area when counting correct label

    Args:
        predicted (torch tensor): _description_
        ssegs (torch tensor): _description_
        meshes (torch tensor): _description_

    Returns:
        correct (int): number of correct segmentation
    """
    correct = 0
    ssegs = ssegs.squeeze(-1) # (B, num_e, num_c, 1) -> (B, num_e, num_c)
    correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2)) # (B, num_e, 1)
    for mesh_id, mesh in enumerate(meshes):
        correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0] # counts real edges
        edge_areas = torch.from_numpy(mesh.get_edge_areas()) # calc edge weight using area
        correct += (correct_vec.float() * edge_areas).sum()
    return correct

def print_network(net):
    """_summary_: Print the total number of parameters in the network
    
    Args:
        net: network with hyper parameters

    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

def get_heatmap_color(value, minimum=0, maximum=1):
    """_summary_

    Args:
        value (int): current value
        minimum (int, optional): _description_. Defaults to 0.
        maximum (int, optional): _description_. Defaults to 1.

    Returns:
        r, g, b: heatmap color of int val
    """
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def normalize_np_array(np_array):
    """_summary_

    Args:
        np_array (ndarray)

    Returns:
        normalized array of value between 0 and 1
    """
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)


def calculate_entropy(np_array):
    """_summary_: Shannon entropy

    Args:
        np_array (ndarray)

    Returns:
        entropy: _description_
    """
    entropy = 0
    np_array /= np.sum(np_array) # normalize into prob
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0]) # range [0,1]
    return entropy
