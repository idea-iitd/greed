from . import config, utils

import torch
from tqdm.auto import tqdm

import heapq
import types
            
class AsymTreeNode:
    def __init__(self, idx, pivot=None, mid1=None, mid2=None, near_near=None, near_far=None, far_near=None, far_far=None):
        self.idx = idx
        self.pivot = pivot
        self.mid1 = mid1
        self.mid2 = mid2
        self.near_near = near_near
        self.near_far = near_far
        self.far_near = far_near
        self.far_far = far_far
            
class AsymTree:        
    @torch.no_grad()
    def __init__(self, tobj, max_leaf_size=1, dist_fn=utils.norm_sed_func):
        tqdm.write(f'construct (metric-)tree for asymmetric/symmetric distance function')
        tqdm.write(f'config.device: {config.device}')
        self.tobj = tobj
        self.idx = torch.arange(len(tobj))
        self.dist_fn = dist_fn
        self.max_leaf_size = max_leaf_size
        self.log = types.SimpleNamespace()
        self.log.n_comp = 0
        self.log.n_inodes = 0
        self.log.n_leaves = 0
        self.log.n_big_leaves = 0
        self.log.max_leaf_size = 0
        self.log.tqdm_bar = tqdm(total=len(tobj))
        self.root = self.build(self.idx)
        self.log.tqdm_bar.close()
        tqdm.write(f'distance computations: {self.log.n_comp}')
        tqdm.write(f'internal nodes: {self.log.n_inodes} / {self.log.n_inodes + self.log.n_leaves}')
        tqdm.write(f'leaf nodes: {self.log.n_leaves} / {self.log.n_inodes + self.log.n_leaves}')
        tqdm.write(f'big leaf nodes: {self.log.n_big_leaves} / {self.log.n_leaves}')
        tqdm.write(f'max leaf size: {self.log.max_leaf_size}')
    
    @torch.no_grad()
    def to(self, device):
        if isinstance(self.tobj, list):
            self.tobj = [x.to(device) for x in self.tobj]
        else:
            self.tobj = self.tobj.to(device)
        self.idx = self.idx.to(config.device)
        return self
        
    def build(self, idx):
        if idx.shape[0] <= self.max_leaf_size:
            self.log.max_leaf_size = max(self.log.max_leaf_size, idx.shape[0])
            self.log.n_leaves += 1
            self.log.tqdm_bar.update(idx.shape[0])
            return AsymTreeNode(idx)
        pivot = idx[0]
        if not isinstance(self.tobj, list):
            dist1 = self.dist_fn(self.tobj[pivot], self.tobj[idx])
        else:
            dist1 = self.dist_fn(self.tobj[pivot], [self.tobj[i] for i in idx])
        self.log.n_comp += idx.shape[0]
        mid1 = torch.median(dist1)
        if not isinstance(self.tobj, list):
            dist2 = self.dist_fn(self.tobj[idx], self.tobj[pivot])
        else:
            dist2 = self.dist_fn(self.tobj[pivot], [self.tobj[i] for i in idx])
        self.log.n_comp += idx.shape[0]
        mid2 = torch.median(dist2)
        mask1 = dist1 <= mid1
        mask2 = dist2 <= mid2
        if torch.all(mask1 & mask2):
            self.log.n_big_leaves += 1
            self.log.max_leaf_size = max(self.log.max_leaf_size, idx.shape[0])
            self.log.n_leaves += 1
            self.log.tqdm_bar.update(idx.shape[0])
            return AsymTreeNode(idx)
        near_near = self.build(idx[mask1 & mask2])
        near_far = self.build(idx[mask1 & ~mask2])
        far_near = self.build(idx[~mask1 & mask2])
        far_far = self.build(idx[~mask1 & ~mask2])
        self.log.n_inodes += 1
        return AsymTreeNode(idx, pivot, mid1, mid2, near_near, near_far, far_near, far_far)
    
    @torch.no_grad()
    def range_query(self, qobj, rlim, slow=False, verbose=False):
        self.qobj = qobj.to(config.device)
        self.rlim = rlim
        self.slow = slow
        self.ret = []
        self.log.n_scan = 0
        self.log.n_comp = 0
        self.log.n_noscan = 0
        self.range_query_aux(self.root)
        if slow:
            self.log.n_ret = len(self.ret)
        else:
            self.ret = torch.cat(self.ret)
            self.log.n_ret = self.ret.shape[0]
        if verbose:
            tqdm.write(f'returned points: {self.log.n_ret} / {self.len(tobj)}')
            tqdm.write(f'scanned points: {self.log.n_scan} / {self.len(tobj)}')
            tqdm.write(f'directly included: {self.log.n_noscan} / {self.len(tobj)}')
            tqdm.write(f'distance computations: {self.log.n_comp}')
            tqdm.write(f'')
        return self.ret
    
    def range_query_aux(self, node):
        if node.pivot is None:
            self.log.n_scan += node.idx.shape[0]
            self.log.n_comp += node.idx.shape[0]
            if self.slow:
                self.ret.extend((i for i in node.idx if self.dist_fn(self.qobj, self.tobj[i]) <= self.rlim))
            else:
                if not isinstance(self.tobj, list):
                    self.ret.append(node.idx[self.dist_fn(self.qobj, self.tobj[node.idx]) <= self.rlim])
                else:
                    self.ret.append(node.idx[self.dist_fn(self.qobj, [self.tobj[i] for i in node.idx]) <= self.rlim])
            return
        d_pq = self.dist_fn(self.tobj[node.pivot], self.qobj)
        d_qp = self.dist_fn(self.qobj, self.tobj[node.pivot])
        self.log.n_comp += 2
        c = d_qp <= self.rlim - node.mid1
        if c:
            self.log.n_noscan += node.near_near.idx.shape[0] + node.near_far.idx.shape[0]
            self.ret.append(node.near_near.idx)
            self.ret.append(node.near_far.idx)
        c1 = d_pq <= node.mid1 - self.rlim
        c2 = d_qp > node.mid2 + self.rlim
        if not (c or c2):
            self.range_query_aux(node.near_near)
        if not c:
            self.range_query_aux(node.near_far)
        if not (c1 or c2):
            self.range_query_aux(node.far_near)
        if not c1:
            self.range_query_aux(node.far_far)
            
    @torch.no_grad()
    def knn_query(self, qobj, k, slow=False, verbose=False):
        self.qobj = qobj.to(config.device)
        self.k = k
        self.heap = [(float('-inf'), None)] * k
        self.rlim = float('inf')
        self.log.n_scan = 0
        self.log.n_comp = 0
        self.slow = slow
        self.knn_query_aux(self.root)
        if verbose:
            tqdm.write(f'scanned points: {self.log.n_scan} / {self.len(tobj)}')
            tqdm.write(f'distance computations: {self.log.n_comp}')
            tqdm.write(f'rank k distance: {self.rlim}')
            tqdm.write(f'')
        return [i for _,i in self.heap]

    def knn_query_aux(self, node):
        if node.pivot is None:
            self.log.n_scan += node.idx.shape[0]
            self.log.n_comp += node.idx.shape[0]
            if self.slow:
                dist = [self.dist_fn(self.qobj, self.tobj[i]) for i in node.idx]
            else:
                if not isinstance(self.tobj, list):
                    dist = self.dist_fn(self.qobj, self.tobj[node.idx])
                else:
                    dist = self.dist_fn(self.qobj, [self.tobj[i] for i in node.idx])
            for i,d in zip(node.idx, dist):
                if d < -self.heap[0][0]:
                    heapq.heappushpop(self.heap, (-d, i))
            self.rlim = -self.heap[0][0]
            return
        d_qp = self.dist_fn(self.qobj, self.tobj[node.pivot])
        d_pq = self.dist_fn(self.tobj[node.pivot], self.qobj)
        self.log.n_comp += 2
        c1 = d_pq <= node.mid1 - self.rlim
        c2 = d_qp >= node.mid2 + self.rlim
        if not c2:
            self.knn_query_aux(node.near_near)
        if True:
            self.knn_query_aux(node.near_far)
        if not (c1 or c2):
            self.knn_query_aux(node.far_near)
        if not c1:
            self.knn_query_aux(node.far_far)
            
class LinearScan:
    @torch.no_grad()
    def __init__(self, tobj, dist_fn=utils.norm_sed_func):
        tqdm.write(f'slow (pythonic) linear scan index')
        tqdm.write(f'config.device: {config.device}')
        self.tobj = tobj
        self.idx = torch.arange(len(tobj))
        self.dist_fn = dist_fn
    
    @torch.no_grad()
    def to(self, device):
        if isinstance(self.tobj, list):
            self.tobj = [x.to(device) for x in self.tobj]
        else:
            self.tobj = self.tobj.to(device)
        self.idx = self.idx.to(config.device)
        return self
    
    @torch.no_grad()
    def range_query(self, qobj, rlim, verbose=True):
        qobj = qobj.to(config.device)
        ret = [i for i, tobj in enumerate(tqdm(self.tobj, 'targets')) if self.dist_fn(qobj, tobj) <= rlim]
        if verbose:
            tqdm.write(f'returned points: {len(ret)}')
            tqdm.write(f'')
        return ret
    
    @torch.no_grad()
    def knn_query(self, qobj, k, verbose=False):
        qobj = qobj.to(config.device)
        heap = [(float('-inf'), None)] * k
        for i, tobj in enumerate(tqdm(self.tobj, 'targets')):
            heapq.heappushpop(heap, (-self.dist_fn(qobj, tobj), i))
        if verbose:
            tqdm.write(f'rank k distance: {-heap[0][0]}')
            tqdm.write(f'')
        return [i for _,i in heap]
    
class FastLinearScan:
    @torch.no_grad()
    def __init__(self, tobj, dist_fn=utils.norm_sed_func):
        tqdm.write(f'fast (vectorised) linear scan index')
        tqdm.write(f'config.device: {config.device}')
        self.tobj = tobj
        self.idx = torch.arange(len(tobj))
        self.dist_fn = dist_fn
    
    @torch.no_grad()
    def to(self, device):
        if isinstance(self.tobj, list):
            self.tobj = [x.to(device) for x in self.tobj]
        else:
            self.tobj = self.tobj.to(device)
        self.idx = self.idx.to(config.device)
        return self
        
    @torch.no_grad()
    def range_query(self, qobj, rlim, verbose=False):
        qobj = qobj.to(config.device)
        ret = self.idx[self.dist_fn(qobj, self.tobj) <= rlim]
        if verbose:
            tqdm.write(f'returned points: {len(ret)}')
            tqdm.write(f'')
        return ret
    
    @torch.no_grad()
    def knn_query(self, qobj, k, verbose=False):
        qobj = qobj.to(config.device)
        _, ret = torch.topk(self.dist_fn(qobj, self.tobj), k, largest=False, sorted=False)
        if verbose:
            tqdm.write(f'rank k distance: <extra computation for unsorted knn query>')
            tqdm.write(f'')
        return ret