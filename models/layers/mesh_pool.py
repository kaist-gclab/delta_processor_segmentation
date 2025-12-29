import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify
from tqdm import tqdm


class MeshPool(nn.Module):
    def __init__(self, target, multi_thread=False):
        """_summary_: edge collapse operator"""
        # target: target edge num
        super(MeshPool, self).__init__()
        self.__out_target = target # output edge dim (int)
        self.__multi_thread = multi_thread # whether or not to use multi-thread (bool)
        self.__fe = None # feature tensor (B, C, e_in, 1)
        self.__updated_fe = None # B, C, e_out, 1)
        self.__meshes = None # list[Mesh] obj
        self.__merge_edges = [-1, -1] # Buffer:[source edge id, target edge id]: current unset

    def __call__(self, fe, meshes):
        # prints log
        # tqdm.write("DEBUG: gemm_edges shape: {}".format(meshes[0].gemm_edges.shape)) # comment out
        # tqdm.write("DEBUG: gemm_edges dtype:".format(meshes[0].gemm_edges.dtype))
        # tqdm.write("DEBUG: gemm_edges[0]:".format(meshes[0].gemm_edges[0]))
        return self.forward(fe, meshes) # log before forward

    def forward(self, fe, meshes):
        """_summary_: pool edge feature to meet self.__out_target edges
        rebuild features"""
        self.__updated_fe = [[] for _ in range(len(meshes))] # gather per mesh output
        pool_threads = [] # thread handling
        self.__fe = fe # (B, C, e_in, 1)
        self.__meshes = meshes # assign meshes (B,)
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                # assign thread to one mesh
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index) # pool
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join() # wait until all pooling finishes
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target) # concat into batch (B, C, target, 1)
        return out_features

    def __pool_main(self, mesh_index):
        """_summary_: pool one mesh"""
        # mesh_index: idx in batch
        # self.__meshes, self.__fe, self.__out_target
        mesh = self.__meshes[mesh_index] # edges_count = e_cur
        # build priority queue: which edge to collapse first
        # heap list of priority val, edge_id
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.edges_count], mesh.edges_count)
        # last_count = mesh.edges_count + 1
        mask = np.ones(mesh.edges_count, dtype=np.bool) # mask in current edge (e_cur,)
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device) # MeshUnion: trasks merge and group
        while mesh.edges_count > self.__out_target: # while edge count bigger
            value, edge_id = heappop(queue) # call edge idx
            edge_id = int(edge_id) # conv to int
            if mask[edge_id]: # if edge alive
                self.__pool_edge(mesh, edge_id, mask, edge_groups) # call pool_edge (edge collapse)
        mesh.clean(mask, edge_groups) # call clean: remap neighbor ref, update edge count, save history
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target) # rebuild feature (1, C, e_out)
        self.__updated_fe[mesh_index] = fe # assign updated fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        """_summary_: collapse single edge"""
        # mesh: mesh being pooled
        # edge_id (int): int edge idx
        # mask (ndarray): current edge mask
        # edge_groups (MeshUnion): merge trasking structure (for feature rebuild)
        if self.has_boundaries(mesh, edge_id): # check if edge is boundary feature
            return False
        # check and cleanup side 0 (one incident face)
        # check and cleanup side 2 (another incident face)
        # one-ring valid - non-manifold(실제 존제 X 경우) / no degeneracy(ill-formed)
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0)\
            and self.__clean_side(mesh, edge_id, mask, edge_groups, 2) \
            and self.__is_one_ring_valid(mesh, edge_id):
            # pool face A, get merged edge id | -1 depend on cond
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
            # pool face B, get merged edge id | -1 depend on cond
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
            mesh.merge_vertices(edge_id) # collapse two vertex of edge id into one vertex
            mask[edge_id] = False # update mask
            MeshPool.__remove_group(mesh, edge_groups, edge_id) # remove edge from edge group
            mesh.edges_count -= 1 # subtract edge count
            return True
        else:
            return False

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        """_summary_: ensure collapsing edge on one face side is safe"""
        if mesh.edges_count <= self.__out_target: # target num meet - return
            return False # 부르는 call을 보면 두개를 동시에 확인해서 상관 X
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side) # list, find invalid local sets of edge_id around side
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target: # invalid edge exist, not target yet
            # triplet: three edge that shares common vertex
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges) # remove invalid triplet
            # after removing, check few cond
            if mesh.edges_count <= self.__out_target: # reach target num
                return False
            if self.has_boundaries(mesh, edge_id): # cleaning caused edge id to become a boundary
                return False # cannot remove edges further
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side) # recompute invalid config after removal
        return True # no invalid remain, safe to collide

    @staticmethod
    def has_boundaries(mesh, edge_id):
        """_summary_: checks if the edge_id is a boundary edge"""
        for edge in mesh.gemm_edges[edge_id]: # edge exists on neighbor edge list
            # print("1: gemm_edge: {}, type: {}".format(mesh.gemm_edges[0], mesh.gemm_edges.dtype))
            if edge == -1 or -1 in mesh.gemm_edges[edge]: # neighbor edge is -1
                return True
        return False


    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        """_summary_: validate one-ring connectivity condition for collapsing edge"""
        # mesh.edges[edge_id, 0]: read v0 | similar for v1
        # mesh.ve[v0]: arrays of incident edges
        # mesh.edges[mesh.ve[v0]]: (deg(v0), 2) > reshape (deg(v0)*2,)
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id]) # from intersection of v0, v1 incident edges - exclude v0 and v1
        return len(shared) == 2 # check incident edge shared (exclude self) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        """_summary_: collapse one face side adjacent to edge_id"""
        # mesh: mesh being pooled
        # edge_id (int): int edge idx
        # mask (ndarray): current edge mask
        # edge_groups (MeshUnion): merge trasking structure (for feature rebuild)
        # side (int): side to collapse
        info = MeshPool.__get_face_info(mesh, edge_id, side) # get local face info
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info # unpack into tuple
        # key_a: edge id to keep, key_b: edge id to remove
        # side_a and side_b: which neighbor slot is involved for current face (side = neighbor edge idx)
        # key_a는 살아있기 때문에 other_side_a, other_key_a는 따로 variable로 안둠 (업뎃 필요 X)
        # other_side_b: pair of two edge id connected to key_b
        # other_keys_b: another pair of two edge id connected to key_b

        # gemm_edges (neighbor update - new edge_id) | sides (new neighbor slot)
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        # redirect key_a even slot(0 or 2) to point key_b's other neighbor (other_keys_b[0])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1], mesh.sides[key_b, other_side_b + 1])
        # redirect key_a odd slot(1 or 3) to point key_b's other neighbor (other_keys_b[0])
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a) # merge key_b into key_a
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a) # merge edge_id into key_a: feature of edge_id will contribute to key_a
        mask[key_b] = False # update mask
        MeshPool.__remove_group(mesh, edge_groups, key_b) # update mesh history, old2current map etc
        mesh.remove_edge(key_b) # safely remove key_b from edges, gemm_edges
        mesh.edges_count -= 1 # decrease count
        return key_a # return surviving edge

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        """_summary_: get invalid local configuration around edge_id"""
        # edge_id (int): int edge idx
        # edge_groups (MeshUnion): union tracker
        # side (int): side to collapse
        info = MeshPool.__get_face_info(mesh, edge_id, side) # get face info
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info # unpack
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b) # find overlap from other neighbor list (index pairs)
        if len(shared_items) == 0: # if no shared item
            return [] # return empty list
        else:
            assert (len(shared_items) == 2) # two list share two item
            middle_edge = other_keys_a[shared_items[0]] # invalid edge - called middle edge | shared edge that lies between both sides of neighbor
            update_key_a = other_keys_a[1 - shared_items[0]] # choose neighbor edge to update
            update_key_b = other_keys_b[1 - shared_items[1]] # same thing for keys_b
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]] # determine corresponding slot idx
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            # Rewire Edge
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a) # rewiring edge_id's neighbor (side)
            MeshPool.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b) # so that no longer point to invalid structure (side+1)
            MeshPool.__redirect_edges(mesh, update_key_a, MeshPool.__get_other_side(update_side_a), update_key_b, MeshPool.__get_other_side(update_side_b)) # connect key_a <-> key_b (otherside slot) - consistency
            # update MeshUnion
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id) # merge key_a and edge_id > edge_id
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id) # merge key_b and edge_id
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a) # merge key_a > update_key_a
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_a) # then merge invalid edge > update_key_a
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b) # merge key_b > update_key_b
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_b) # merge invalid edge > update_key_b
            return [key_a, key_b, middle_edge] # return triplet to remove

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        """_summary_: rewire edge-neighborhood adjacency between two edges"""
        # link: (edge_a_key, side_a)  <-->  (edge_b_key, side_b)
        # edge keys: edge id | side: slot index
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key # edge_a's neighbor of slot a = edge_b
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key # edge_b's neighbor of slot b = edge_a
        mesh.sides[edge_a_key, side_a] = side_b # edge_b > edge_a = uses slot b
        mesh.sides[edge_b_key, side_b] = side_a # edge_a > edge_b = uses slot a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        """_summary_: matching item from two lists"""
        # usually two lists are edges (len 2)
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]: # same comp
                    shared_items.extend([i, j]) # extend pair of index map to same comp
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        """_summary_: local edge-neighborhood info afound face"""
        # edge_id (int): int edge idx
        # side (int): side idx
        key_a = mesh.gemm_edges[edge_id, side] # neighbor edge id (side)
        key_b = mesh.gemm_edges[edge_id, side + 1] # neighbor edge id (side+1)
        side_a = mesh.sides[edge_id, side] # backtracking idx key_a to edge id
        side_b = mesh.sides[edge_id, side + 1] # backtracking idx key_b to edge id
        other_side_a = (side_a - (side_a % 2) + 2) % 4 # 0 or 2 (other side)
        other_side_b = (side_b - (side_b % 2) + 2) % 4 # 1 or 3 (other side)
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]] # two neighbor edges of key_a on other face pair
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]] # two neighbor edges of key_b on other face pair
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b # return all retrieved info

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        """_summary_: remove triplet - three edge share single vertex"""
        vertex = set(mesh.edges[invalid_edges[0]]) # first edge's vertex
        for edge_key in invalid_edges: # for each edge in triplet
            vertex &= set(mesh.edges[edge_key]) # get intersection with set - with all edges vertex set
            mask[edge_key] = False # mark edge as removed
            MeshPool.__remove_group(mesh, edge_groups, edge_key) # remove group from MeshUnion / Mesh
        mesh.edges_count -= 3 # update counter
        vertex = list(vertex) # convert into list: this would contain exactly one vertex
        assert(len(vertex) == 1) # check if the intersection is one vertex
        mesh.remove_vertex(vertex[0]) # remove common vertex from Mesh

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        """_summary_: get and update union groups"""
        edge_groups.union(source, target) # get union group | update current level pooling
        mesh.union_groups(source, target) # assugn to union_groups

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        """_summary_: remove edge from current level feature and history"""
        edge_groups.remove_group(index) # update group: call MeshUnion remove group
        mesh.remove_group(index) # update mesh history: call Mesh remove group (edge mask, old2current from mesh)