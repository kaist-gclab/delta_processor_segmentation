from tempfile import mkstemp
from shutil import move
import torch
import numpy as np
import os
from models.layers.mesh_union import MeshUnion
from models.layers.mesh_prepare import fill_mesh


class Mesh:

    def __init__(self, file=None, opt=None, hold_history=False, export_folder=''):
        # vs: vertex position (v_num, 3)
        # v_mask: vertex mask (v_num)
        # features: edge features (C, num_e) or (C, num_e, 1)
        # edge_areas: per-edge area weight (num_e,)
        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        # membership / neighbor related
        # edges: vertex idx of edge (e_num, 2)
        # gemm_edges: 1-ring neighbor (e_num, 4)
        # sides: opposite / neighbor reading dir - idx (e_num, 4)
        self.edges = self.gemm_edges = self.sides = None
        self.pool_count = 0
        # initialized by fill_mesh
        # edge_count (int)
        # ve (dict): key - vertex, val - incident edge
        fill_mesh(self, file, opt) # file: mesh file path
        self.export_folder = export_folder # export(), export_segments(), history_data['collapses']
        self.history_data = None # enable init_history() to store pool/unpool info
        # IO-related
        # filename: mesh filename
        # export_folder: output dir string / None-return early
        # initialized by init_history
        # history_data: dict stores pool/unpool history
        if hold_history: # bool
            self.init_history()
        self.export()

    def extract_features(self):
        """_summary_: returns features"""
        return self.features

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        concat = np.concatenate([self.ve[edge[0]], self.ve[edge[1]]])
        self.ve[edge[0]] = concat
        # self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            # find index of edge_id & remove
            if len(self.ve[v]) > 0:
                vidx = np.where(self.ve[v] == edge_id)
                removed = np.delete(self.ve[v], vidx)
                self.ve[v] = removed
            # self.ve[v].remove(edge_id)

    def clean(self, edges_mask, groups):
        # edge_mask: bool array, current edge b4 cleaning (e_old,)
        # groups: MeshUnion instance (update occurrences, groups)
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy()) # convert mask
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = [] # using vertex idx, find incident edge (nested list, ragged)
        edges_mask = np.concatenate([edges_mask, [False]]) # (e_old + 1, ) False slot never kept - padding
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32) # remapping edge idx
        new_indices[-1] = -1 # set dummy idx as -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0]) # assign new id (for alive edges)
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]] # remaps edge neighbors
        for v_index, ve in enumerate(self.ve):
            update_ve = [] # stores new edge idx
            # if self.v_mask[v_index]:
            for e in ve: # all prev edge idx
                update_ve.append(new_indices[e]) # add newly assigned edge idx to edge neighbor
            new_ve.append(update_ve)
        self.ve = new_ve # assign new mapping
        self.__clean_history(groups, torch_mask) # record variable edge-related arguments
        self.pool_count += 1
        self.export()


    def export(self, file=None, vcolor=None):
        """_summary_: export info to dir"""
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename) # split basename, extension
                file = '%s/%s_%d%s' % (self.export_folder, filename, self.pool_count, file_extension)
                # export_folder/fname_poolcnt.extension
            else:
                return
        faces = []
        vs = self.vs[self.v_mask] # alive vertex position (mask applied)
        gemm = np.array(self.gemm_edges) # gather 1-ring neighbor
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32) # initialize new vertex indices
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0]) # save alive vidx
        for edge_index in range(len(gemm)):
            cycles = self.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                faces.append(self.__cycle_to_face(cycle, new_indices))
        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in self.edges:
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            edge_key = 0
            with os.fdopen(fh, 'w') as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == 'e':
                            new_file.write('%s %d' % (line.strip(), cur_segments[edge_key]))
                            if edge_key < len(cur_segments):
                                edge_key += 1
                                new_file.write('\n')
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['edges_mask']):
                cur_segments = segments[:len(self.history_data['edges_mask'][i])]
                cur_segments = cur_segments[self.history_data['edges_mask'][i]]

    def __get_cycle(self, gemm, edge_id):
        """_summary_: walks upon gemm and creates two cycle"""
        # gemm (e_num,4), edge_id (int)
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2 # j=0 > 0 / j=1 > 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1: # no neighbor
                continue # skip
            cycles.append([]) # append empty lst
            for i in range(3): # three edge
                tmp_next_key = gemm[next_key, next_side] # get next neighbor to visit
                tmp_next_side = self.sides[next_key, next_side] # calculate side(of current move) corresponds to neighbor
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2) # even: add 1, odd: subtrace 1 | 0<>1, 2<>3
                gemm[next_key, next_side] = -1 # mask visited
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1 # same face, disable
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key) # append the current edge
        return cycles # up to 2 cycle, each - list of edge id (three)

    def __cycle_to_face(self, cycle, v_indices):
        """_summary_: calculate face (vertex idx) from three edge"""
        # cycle (3,): e0, e1, e2 | v_indices (v_num)
        face = []
        for i in range(3):
            # 두 개의 edge에 포함된 하나의 vertex를 구함
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face # (3,)

    def init_history(self):
        # exist per mesh instance / updated when pool/unpool
        # groups: matrices produced at pooling (e_in, unroll_target)
        # gemm_edges: snapshot of gemm_edges(neighbor) per pooling lvl
        # occurrences: occurrences saved at pooling step (e_new,)
        # old2current: map original edge id to current edge id (e_num,)
            # -1 if X exist
        # current2old: map current edge id to old edge id (e_num,)
        # edges_mask: bool val, trace if edge still exist on current pooling level (e_num,)
        # edges_count: int val, scalar edge count per level
        self.history_data = {
                               'groups': [],
                               'gemm_edges': [self.gemm_edges.copy()],
                               'occurrences': [],
                               'old2current': np.arange(self.edges_count, dtype=np.int32),
                               'current2old': np.arange(self.edges_count, dtype=np.int32),
                               'edges_mask': [torch.ones(self.edges_count,dtype=torch.bool)],
                               'edges_count': [self.edges_count],
                              }
        # collapses: MeshUnion obj - track union / if export folder exists
        if self.export_folder:
            self.history_data['collapses'] = MeshUnion(self.edges_count)

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data['collapses'].union(self.history_data['current2old'][source], self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['edges_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            if self.export_folder:
                self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()
    
    def __clean_history(self, groups, pool_mask):
        """_summary_: update edge-rel features"""
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(self.edges_count, dtype=np.int32)
            self.history_data['current2old'][0: self.edges_count] = np.ma.where(mask)[0]
            if self.export_folder != '':
                self.history_data['edges_mask'].append(self.history_data['edges_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)
    
    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]

    def get_edge_areas(self):
        return self.edge_areas
