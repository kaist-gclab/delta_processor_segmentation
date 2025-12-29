import torch
import torch.nn as nn


class MeshUnpool(nn.Module):
    """_summary_: undo pooling in feature space"""
    def __init__(self, unroll_target):
        super(MeshUnpool, self).__init__()
        self.unroll_target = unroll_target

    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group, unroll_start):
        """_summary_: pad group dim to unroll_start
        so batching works"""
        # group (E_in, unroll_target)
        start, end = group.shape
        padding_rows =  unroll_start - start # padding row dim
        padding_cols = self.unroll_target - end # padding col dim
        # padding
        if padding_rows != 0 or padding_cols !=0:
            padding = nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            group = padding(group)
        return group # (unroll_start, unroll_target)

    def pad_occurrences(self, occurrences):
        """_summary_: pad member count(occurences) vector to unroll target"""
        # occurrences (E_out_mesh)
        padding = self.unroll_target - occurrences.shape[0] # padding dim
        # padding
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences # (unroll_target)

    def forward(self, features, meshes):
        """_summary_: build padded group matrix for the batch

        Args:
            features (_type_): _description_
            meshes (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size, nf, edges = features.shape
        groups = [self.pad_groups(mesh.get_groups(), edges) for mesh in meshes] # (E_in, unroll_target)
        # unroll_mat: maps current edges to unrolled edges
        unroll_mat = torch.cat(groups, dim=0).view(batch_size, edges, -1) # (B, E_in, unroll_target)
        # pad occurances
        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes] # mesh.get_occ (unroll_target)
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1) # (B, 1, unroll_target)
        occurrences = occurrences.expand(unroll_mat.shape) # (B, E_in, unroll_target) > match unroll_mat
        unroll_mat = unroll_mat / occurrences
        unroll_mat = unroll_mat.to(features.device)
        for mesh in meshes:
            mesh.unroll_gemm()
        return torch.matmul(features, unroll_mat)
