import torch
from torch.nn import ConstantPad2d


class MeshUnion:
    """_summary_: group of edge and its functions"""
    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n # n group - one per edge
        self.rebuild_features = self.rebuild_features_average
        self.groups = torch.eye(n, device=device) # (n, n) membership matrix

    def union(self, source, target):
        """_summary_: adds membership to target"""
        self.groups[target, :] += self.groups[source, :]

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        """_summary_: returns membership group"""
        return self.groups[edge_key, :]

    def get_occurrences(self):
        """_summary_: for each edge, how many groups"""
        return torch.sum(self.groups, 0) # (n, )

    def get_groups(self, tensor_mask):
        """_summary_: membership value to binary 0/1"""
        self.groups = torch.clamp(self.groups, 0, 1)
        # tensor mask selects m group
        return self.groups[tensor_mask, :] # (m, n)

    def rebuild_features_average(self, features, mask, target_edges):
        """_summary_: re-build pooled edge features"""
        # feature: (B, C, E, 1), mask (E,), target edges(int): desired edge count
        self.prepare_groups(features, mask) # self groups shaped matmul
        # aggreate feature by group membership
        fe = torch.matmul(features.squeeze(-1), self.groups) # (B, C, E_new)
        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        fe = fe / occurrences # average (divide by member count)
        # padding
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe # (B, C, target_edge)

    def prepare_groups(self, features, mask):
        """_summary_: reshape membership feature to match edge mask"""
        tensor_mask = torch.from_numpy(mask)
        self.groups = torch.clamp(self.groups[tensor_mask, :], 0, 1).transpose_(1, 0)
        padding_a = features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)
