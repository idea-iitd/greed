import sys
sys.path.insert(0, '../pyged/lib')
import pyged
from . import config, utils, viz

import numpy as np
import torch
import torch_geometric as tg
import torch_geometric.data
import torch_geometric.datasets
from tqdm.auto import tqdm

import networkx as nx

import heapq
import itertools as it
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import os
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def k_hop_nbr_nx(nx_g, src, n_hops):
    return tg.utils.from_networkx(nx.convert_node_labels_to_integers(nx.ego_graph(nx_g, src, n_hops)))

def random_bfs_sample(g, n_hops, trav_prob, node_mask=None, edge_mask=None):
    if node_mask is None or torch.sum(node_mask) == 0:
        front_mask = torch.zeros(g.num_nodes, dtype=torch.bool)
        front_mask[0] = True
        seen_node_mask = torch.zeros(g.num_nodes, dtype=torch.bool)
        seen_node_mask[0] = True
    else:
        front_mask = node_mask.detach().clone().cpu()
        seen_node_mask = node_mask.detach().clone().cpu()
    if edge_mask is None:
        seen_edge_mask = torch.zeros(g.num_edges, dtype=torch.bool)
    else:
        seen_edge_mask = edge_mask.detach().clone().cpu()
    for _ in range(n_hops):
        random_mask = torch.rand(g.num_edges) <= trav_prob
        edge_mask = front_mask[g.edge_index[0]] & random_mask
        seen_edge_mask |= edge_mask
        front_mask.fill_(False)
        front_mask[g.edge_index[1, edge_mask]] = True
        front_mask &= ~seen_node_mask
        seen_node_mask |= front_mask
    new_x = g.x[seen_node_mask]
    new_index = torch.empty(g.num_nodes, dtype=torch.long)
    new_index[seen_node_mask] = torch.arange(torch.sum(seen_node_mask))
    new_edge_index = new_index[g.edge_index[:, seen_edge_mask]]
    new_edge_index = torch.unique(torch.cat((new_edge_index, new_edge_index[[1,0],:]), dim=1), dim=1)
    return tg.data.Data(new_x, new_edge_index)

nx_g, n_hops = None, None
def k_hop_nbr_init(*args):
    global nx_g, n_hops
    graph, n_hops = args
    nx_g = tg.utils.to_networkx(graph, node_attrs=['x'], to_undirected=True, remove_self_loops=False)
    
def k_hop_nbr_func(args):
    src = args
    return src, k_hop_nbr_nx(nx_g, src, n_hops)

def decompose(graphs, n_hops):
    tqdm.write('decompose into neighborhoods')
    ret = []
    for graph in tqdm(graphs, desc='graphs'):
        r = [None] * graph.num_nodes
        tqdm.write(f'n_workers: {config.n_workers}')
        if config.n_workers == 1:
            k_hop_nbr_init(graph, n_hops)
            for src, nbr in tqdm(map(k_hop_nbr_func, range(graph.num_nodes)), desc='nbrs', total=graph.num_nodes):
                r[src] = nbr
        else:
            with mp.Pool(config.n_workers, k_hop_nbr_init, (graph, n_hops)) as p:
                for src, nbr in tqdm(p.imap_unordered(k_hop_nbr_func, range(graph.num_nodes)), desc='nbrs', total=graph.num_nodes):
                    r[src] = nbr
        ret += r
    return ret

p_queries, p_targets = None, None
def sed_init(*args):
    global p_queries, p_targets
    p_queries, p_targets = args

def sed_func(args):
    i, j = args
    return i, j, pyged.sed(p_queries[i], p_targets[j], config.method_name, config.method_args)

def sed_plus_func(args):
    i, j = args
    return i, j, pyged.sed_plus(p_queries[i], p_targets[j], config.method_name, config.method_args)

def inner_sed(queries, targets):
    tqdm.write('compute inner sed (bounds only)')
    assert len(queries) == len(targets)
    n = len(queries)
    lb = torch.empty(n)
    ub = torch.empty(n)
    p_queries = [utils.to_pyged(query) for query in queries]
    p_targets = [utils.to_pyged(target) for target in targets]
    
    if config.n_workers == 1:
        sed_init(p_queries, p_targets)
        tqdm.write(f'n_workers: {config.n_workers}')
        tqdm.write(f'method_name: {config.method_name}')
        tqdm.write(f'method_args: {config.method_args}')
        for i, j, sed_data in \
        tqdm(map(sed_func, ((i,i) for i in range(n))), desc='pairs', total=n):
            lb[i], ub[j] = sed_data
    else:
        with mp.Pool(config.n_workers, sed_init, (p_queries, p_targets)) as p:
            tqdm.write(f'n_workers: {config.n_workers}')
            tqdm.write(f'method_name: {config.method_name}')
            tqdm.write(f'method_args: {config.method_args}')
            for i, j, sed_data in \
            tqdm(p.imap_unordered(sed_func, ((i,i) for i in range(n))), desc='pairs', total=n):
                lb[i], ub[j] = sed_data

    return lb, ub

def inner_sed_plus(queries, targets):
    tqdm.write('compute inner sed (bounds + nearest subgraph)')
    assert len(queries) == len(targets)
    n = len(queries)
    lb = torch.empty(n)
    ub = torch.empty(n)
    node_masks = [None]*n
    edge_masks = [None]*n
    p_queries = [utils.to_pyged(query) for query in queries]
    p_targets = [utils.to_pyged(target) for target in targets]
    
    if config.n_workers == 1:
        sed_init(p_queries, p_targets)
        tqdm.write(f'n_workers: {config.n_workers}')
        tqdm.write(f'method_name: {config.method_name}')
        tqdm.write(f'method_args: {config.method_args}')
        for i, j, sed_data in \
        tqdm(map(sed_plus_func, ((i,i) for i in range(n))), desc='pairs', total=n):
            lb[i], ub[j], node_mask_idx, edge_mask_idx = sed_data
            node_masks[i] = torch.zeros(targets[i].num_nodes, dtype=torch.bool)
            node_masks[i][node_mask_idx] = True
            edge_masks[j] = torch.zeros(targets[j].num_edges, dtype=torch.bool)
            edge_masks[j][edge_mask_idx] = True
    else:
        with mp.Pool(config.n_workers, sed_init, (p_queries, p_targets)) as p:
            tqdm.write(f'n_workers: {config.n_workers}')
            tqdm.write(f'method_name: {config.method_name}')
            tqdm.write(f'method_args: {config.method_args}')
            for i, j, sed_data in \
            tqdm(p.imap_unordered(sed_plus_func, ((i,i) for i in range(n))), desc='pairs', total=n):
                lb[i], ub[j], node_mask_idx, edge_mask_idx = sed_data
                node_masks[i] = torch.zeros(targets[i].num_nodes, dtype=torch.bool)
                node_masks[i][node_mask_idx] = True
                edge_masks[j] = torch.zeros(targets[j].num_edges, dtype=torch.bool)
                edge_masks[j][edge_mask_idx] = True

    return lb, ub, node_masks, edge_masks

def outer_sed(queries, targets):
    tqdm.write('compute outer sed')
    nq = len(queries)
    nt = len(targets)
    lb = torch.empty((nq, nt))
    ub = torch.empty((nq, nt))
    p_queries = [utils.to_pyged(query) for query in queries]
    p_targets = [utils.to_pyged(target) for target in targets]
    
    if config.n_workers == 1:
        sed_init(p_queries, p_targets)
        tqdm.write(f'n_workers: {config.n_workers}')
        tqdm.write(f'method_name: {config.method_name}')
        tqdm.write(f'method_args: {config.method_args}')
        for i, j, sed_data in \
        tqdm(map(sed_func, it.product(range(nq), range(nt))), desc='pairs', total=nq*nt):
            lb[i,j], ub[i,j] = sed_data
    else:
        with mp.Pool(config.n_workers, sed_init, (p_queries, p_targets)) as p:
            tqdm.write(f'n_workers: {config.n_workers}')
            tqdm.write(f'method_name: {config.method_name}')
            tqdm.write(f'method_args: {config.method_args}')
            for i, j, sed_data in \
            tqdm(p.imap_unordered(sed_func, it.product(range(nq), range(nt))), desc='pairs', total=nq*nt):
                lb[i,j], ub[i,j] = sed_data

    return lb, ub

targets, n_hops, trav_prob = None, None, None
def query_init(*args):
    global targets, n_hops, trav_prob
    targets, n_hops, trav_prob = args

def query_func(args):
    while True:
        query = random_bfs_sample(random.choice(targets), n_hops, trav_prob)
        if query.num_edges >= 1:
            break
    return query

def make_queries(targets, n_queries, n_hops, trav_prob, node_lim=None):
    tqdm.write('sample queries from targets')
    ret = []
    # parallel version of this gets stuck after a certain number of calls to make_queries for some reason
#     n = n_queries
#     with mp.Pool(config.n_workers, query_init, (targets, n_hops, trav_prob)) as p:
#         for q in tqdm(p.imap_unordered(query_func, it.repeat(None, n)), total=n, desc='sampled queries'):
#             ret.append(q)
    for _ in tqdm(range(n_queries), desc='sampled queries'):
        while True:
            query = random_bfs_sample(random.choice(targets), n_hops, trav_prob)
            if (query.num_edges >= 1) and (query.num_nodes >= 5) and (node_lim is None or query.num_nodes <= node_lim):
                break
        ret.append(query)
    return ret

def make_inner_dataset(graphs, n_pairs, n_hops_query, trav_prob_query, node_lim_query=None, n_hops_target=None, targets=None):
    tqdm.write('make inner dataset')
    if targets is None:
        targets = decompose(graphs, n_hops_target) if n_hops_target else graphs
    queries = make_queries(targets, n_pairs, n_hops_query, trav_prob_query, node_lim_query)
    targets = random.choices(targets, k=n_pairs)
    lb, ub = inner_sed(queries, targets)
    return queries, targets, lb, ub

def join_inner_datasets(datasets):
    l_queries, l_targets, l_lb, l_ub = zip(*datasets)
    queries = list(it.chain.from_iterable(l_queries))
    targets = list(it.chain.from_iterable(l_targets))
    lb = torch.cat(l_lb)
    ub = torch.cat(l_ub)
    return queries, targets, lb, ub

def make_inner_dataset_plus(graphs, n_pairs, n_hops_query, trav_prob_query, node_lim_query=None, n_hops_target=None, targets=None):
    tqdm.write('make inner dataset')
    if targets is None:
        targets = decompose(graphs, n_hops_target) if n_hops_target else graphs
    queries = make_queries(targets, n_pairs, n_hops_query, trav_prob_query, node_lim_query)
    targets = random.choices(targets, k=n_pairs)
    lb, ub, node_masks, edge_masks = inner_sed_plus(queries, targets)
    return (queries, targets, lb, ub), (node_masks, edge_masks)

def make_outer_dataset(graphs, n_queries, n_hops_query, trav_prob_query, node_lim_query=None, n_hops_target=None, n_targets=None):
    tqdm.write('make outer dataset')
    targets = decompose(graphs, n_hops_target) if n_hops_target else graphs
    targets = random.sample(targets, n_targets) if n_targets else targets
    queries = make_queries(targets, n_queries, n_hops_query, trav_prob_query, node_lim_query)
    lb, ub = outer_sed(queries, targets)
    return queries, targets, lb, ub

def make_target_transform(n_hops):
    def target_transform(arg):
        item, meta = arg
        query, target, lb, ub = item
        node_mask, edge_mask = meta
#         if random.random() >= 0.5 and n_hops != 0:
#             new_target = target
#         else:
        if True:
            new_target = random_bfs_sample(target, 0 if n_hops == 0 else random.randint(1, n_hops), random.random(), node_mask, edge_mask)
        return query, new_target, lb, ub
    
    return target_transform

class AugmentedInnerDataset(tg.data.Dataset):
    def __init__(self, dataset, metaset, n_hops):
        super().__init__(transform=make_target_transform(n_hops))
        self.queries, self.targets, self.lb, self.ub = dataset
        self.node_masks, self.edge_masks = metaset
    
    def __len__(self):
        return len(self.queries)
    
    def get(self, i):
        return ((self.queries[i], self.targets[i], self.lb[i], self.ub[i]),
                (self.node_masks[i], self.edge_masks[i]))

def impute(g):
    if g['x'] is None:
        g['x'] = torch.ones((g.num_nodes, 1))
    return g

pre_filter = lambda g: g.num_nodes <= 10

def build_sed_dataset(name):    
    tg_root = os.path.join('../data/tg', name)
    train_graphs = [utils.to_pyged(impute(g)) for g in tg.datasets.GEDDataset(tg_root, name, train=True, pre_filter=pre_filter)]
    test_graphs = [utils.to_pyged(impute(g)) for g in tg.datasets.GEDDataset(tg_root, name, train=False, pre_filter=pre_filter)]
    
    sed_root = os.path.join('../data/sed', name)
    try:
        os.makedirs(sed_root)
    except FileExistsError:
        pass
    with open(os.path.join(sed_root, 'train.csv'), 'w') as train_file:
        for g in tqdm(train_graphs, desc='train.csv'):
            for h in train_graphs:
                lb, ub = pyged.sed_bounds(g, h)
                train_file.write(f'{lb},{ub}\n')
                train_file.flush()
    
    with open(os.path.join(sed_root, 'test.csv'), 'w') as test_file:
        for g in tqdm(test_graphs, desc='test.csv'):
            for h in train_graphs:
                lb, ub = pyged.sed_bounds(g, h)
                test_file.write(f'{lb},{ub}\n')
                test_file.flush()

class SEDDataset(tg.data.Dataset):
    def __init__(self, name, train):
        super().__init__()
        tg_root = os.path.join('../data/tg', name)
        train_graphs = [impute(g) for g in tg.datasets.GEDDataset(tg_root, name, train=True, pre_filter=pre_filter)]
        test_graphs = [impute(g) for g in tg.datasets.GEDDataset(tg_root, name, train=False, pre_filter=pre_filter)]
        self.g = train_graphs if train else test_graphs
        self.h = train_graphs
        sed_root = os.path.join('../data/sed', name)
        sed_path = os.path.join(sed_root, 'train.csv' if train else 'test.csv')
        sed = np.genfromtxt(sed_path, delimiter=',')
        self.lb = torch.tensor(sed[:,0])
        self.ub = torch.tensor(sed[:,1])
    
    def to(self, device):
        self.g = [x.to(device) for x in self.g]
        self.h = [x.to(device) for x in self.h]
        self.lb = self.lb.to(device)
        self.ub = self.ub.to(device)
        return self
    
    def __len__(self):
        return len(self.g) * len(self.h)

    def get(self, idx):
        i = idx // len(self.h)
        j = idx % len(self.h)
        return self.g[i], self.h[j], self.lb[idx], self.ub[idx]

class OTFDataset(torch.utils.data.IterableDataset):
    def __init__(self, lb_only=False):
        super().__init__()
        dataset = tg.datasets.CitationFull(root='../data/tg/Cora_ML', name='Cora_ML')
        target = dataset[0]
        n_classes = torch.max(target.y)+1
        target.x = torch.zeros((target.y.shape[0], n_classes))
        target.x[torch.arange(n_classes) == target.y[:,None]] = 1
        target.y = None
        #self.nbrs1 = [utils.k_hop_nbr(target, i, 1) for i in tqdm(range(target.num_nodes))]
        self.nbrs2 = [utils.k_hop_nbr(target, i, 2) for i in tqdm(range(target.num_nodes))]
        self.nbrs2 = [g for g in self.nbrs2 if g.num_nodes <= 100]
        #self.nbrs3 = [utils.k_hop_nbr(target, i, 3) for i in tqdm(range(target.num_nodes))]
        self.lb_only = lb_only
        
    def gen(self):
        while True:
            g = random.choice(self.nbrs2)
            h = random.choice(self.nbrs2)
            lb, ub = pyged.sed_bounds(utils.to_pyged(g), utils.to_pyged(h))
            if self.lb_only:
                ub = lb
            yield g, h, lb, ub

    def __iter__(self):
        return self.gen()
    
def load_graph_from_txt(path):
    labels = []
    with open(os.path.join(path, 'labels.txt')) as f:
        for l in tqdm(f, 'labels'):
            labels.append(int(l.split()[1]))
    edges = []
    with open(os.path.join(path, 'edges.txt')) as f:
        for l in tqdm(f, 'edges'):
            edges.append(list(map(int, l.split())))
    x = torch.tensor(labels).view(-1,1)
    edge_index = torch.tensor(edges).t()
    return tg.data.Data(x, edge_index)

def load_amazon_from_txt(input_file):
    edges_src,edges_dst = [],[]
    nodes = set()
    with open(input_file,mode='rb') as gin:
        for iline,line in enumerate(gin.readlines()):
            line = line.decode('ascii')
            if line[0].isnumeric():
                toks = line.split("\t")
                src = int(toks[0])
                nodes.add(src)
                for tok in toks[1:]:
                    dst = int(tok)
                    nodes.add(dst)
                    edges_src.append(src)
                    edges_dst.append(dst)
    nodes = list(nodes)
    nodes.sort()
    node_map = dict()
    for i in range(len(nodes)):
        node_map[nodes[i]] = i
    edges_src = [node_map[e] for e in edges_src]
    edges_dst = [node_map[e] for e in edges_dst]
    edge_index = tg.utils.to_undirected(torch.tensor([edges_src,edges_dst], dtype=torch.long))
    x = torch.tensor([[1,0] for _ in range(len(nodes))], dtype=torch.float)
    return [tg.data.Data(x=x, edge_index=edge_index)]

def load_ubuntu_from_txt(input_file):
    edges_src,edges_dst = [],[]
    nodes,edges = set(),set()
    with open(input_file,mode='r') as gin:
        for line in gin.readlines():
                toks = line.split(" ")
                src,dst = int(toks[0]),int(toks[1])
                nodes.add(src)
                nodes.add(dst)
                edges.add((src,dst))
    nodes = list(nodes)
    nodes.sort()
    node_map = dict()
    for i in range(len(nodes)):
        node_map[nodes[i]] = i
    edges_src = [node_map[s] for (s,d) in edges]
    edges_dst = [node_map[d] for (s,d) in edges]
    edge_index = tg.utils.to_undirected(torch.tensor([edges_src,edges_dst], dtype=torch.long))
    edge_index,_ = tg.utils.remove_self_loops(edge_index)
    x = torch.tensor([[1,0] for _ in range(len(nodes))], dtype=torch.float)
    return [tg.data.Data(x=x, edge_index=edge_index)]

def load_dblp_targets(input_file, label_path, num_classes=10):
    targets,labels = [],[]
    with open(label_path,mode='r') as gin:
        for line in gin.readlines():
            line = line.strip()
            if(len(line)>0):
                toks = line.split(' ')
                labels.append(int(toks[1]))
    
    print(len(labels))
    
    edges_src,edges_dst = [],[]
    nodes = set()
    with open(input_file,mode='r') as gin:
        flag,count = 0,0
        pbar = tqdm(total=3177888)
        while(1):
            line = gin.readline()
            if(not line):
                break
            line = line.strip()
            if(flag == 0):
                if(len(line) > 0):
                    flag = 1
                else:
                    continue
            if(flag == 1):
                if(len(line) == 0):
                    count += 1
                    nodes = list(nodes)
                    nodes.sort()
                    node_map = dict()
                    for i in range(len(nodes)):
                        node_map[nodes[i]] = i
                    edges_src = [node_map[e] for e in edges_src]
                    edges_dst = [node_map[e] for e in edges_dst]
                    edge_index = tg.utils.to_undirected(torch.tensor([edges_src,edges_dst], dtype=torch.long))
                    x = torch.nn.functional.one_hot(torch.tensor([labels[n]%num_classes for n in nodes], dtype=torch.int64),num_classes).float()
                    targets.append(tg.data.Data(x=x, edge_index=edge_index))
                    flag = 0
                    edges_src,edges_dst = [],[]
                    nodes = set()
                    pbar.update(1)
                else:
                    toks = line.split(" ")
                    src = int(toks[0])
                    nodes.add(src)
                    dst = int(toks[1])
                    nodes.add(dst)
                    edges_src.append(src)
                    edges_dst.append(dst)
        pbar.close()
    
    return targets

def load_proteins_full(min_num_nodes = 2, max_num_nodes = 2000, path = '../data/tg/PROTEINS_full/',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    name = 'PROTEINS_full'
    tqdm.write('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    tqdm.write('Loaded')
    for i in range(len(graphs)):
        graphs[i] = tg.utils.from_networkx(graphs[i])
        graphs[i].y = graphs[i].label
        del graphs[i].label
        
    return graphs

class GEDDataset(tg.data.Dataset):
    def __init__(self, queries, targets):
        super().__init__()
        self.queries = queries
        self.targets = targets
    
    def __len__(self):
        return len(self.queries) * len(self.targets)
    
    def get(self, idx):
        i = idx // len(self.targets)
        j = idx % len(self.targets)
        q = self.queries[i]
        t = self.targets[j]
        ged = self.queries.ged[q.i, t.i]
        if q['x'] is None:
            q['x'] = torch.ones(q.num_nodes,1)
        if t['x'] is None:
            t['x'] = torch.ones(t.num_nodes,1)
        return (q, t, ged.item(), ged.item())
    
def topk_seds_aux(k, query, targets, lbs, ubs):
    p_query = utils.to_pyged(query)
    idx = [i for i in range(len(targets))]
    idx = sorted(idx, key=lambda i: (lbs[i], -ubs[i]))
    targets = [targets[r] for r in idx]
    lbs = [lbs[r] for r in idx]
    ubs = [ubs[r] for r in idx]
    buf = []
    heap = [(float('-inf'), None)] * k
    for i, target in enumerate(tqdm(targets, 'targets', leave=False)):
        if lbs[i] > -heap[0][0]:
            break
        if lbs[i] == ubs[i]:
            sed = lbs[i]
        else:
            p_target = utils.to_pyged(target)
            lb, ub = pyged.sed(p_query, p_target, config.method_name, config.method_args)
            sed = (lb+ub)/2
            if lb != ub:
                print(f'NOT EXACT: lb={lb} ub={ub}')
        pn_sed, pidx = heapq.heappushpop(heap, (-sed, idx[i]))
        if pn_sed == heap[0][0]:
            if buf and pn_sed == buf[0][0]:
                buf.append((pn_sed, pidx))
            else:
                buf = [(pn_sed, pidx)]
    seds = [None] * k
    idxs = [None] * k
    for i in range(k):
        seds[i], idxs[i] = heapq.heappop(heap)
        seds[i] = -seds[i]
    seds = list(reversed(seds))
    idxs = list(reversed(idxs))
    seds += [None] * len(buf)
    idxs += [None] * len(buf)
    for i in range(len(buf)):
        seds[k+i], idxs[k+i] = buf[i]
        seds[k+i] = -seds[k+i]
    return seds, idxs
    
k, targets = None, None
def topk_seds_init(*args):
    global k, targets
    k, targets = args
    
def topk_seds_func(args):
    i, query, lbs, ubs = args
    return i, topk_seds_aux(k, query, targets, lbs, ubs)
    
def topk_seds(k, outer_dataset):
    tqdm.write('exact topk seds')
    queries, targets, lb, ub = outer_dataset
    lbs = lb.tolist()
    ubs = ub.tolist()
    seds = [None] * len(queries)
    idxs = [None] * len(queries)
    if config.n_workers == 1:
        for i, query in enumerate(tqdm(queries, 'queries')):
            seds[i], idxs[i] = topk_seds_aux(k, query, targets, lbs[i], ubs[i])
    else:
        tqdm.write(f'config.n_workers: {config.n_workers}')
        tqdm.write(f'config.method_name: {config.method_name}')
        tqdm.write(f'config.method_args: {config.method_args}')
        with mp.Pool(config.n_workers, topk_seds_init, (k, targets)) as p:
            for i, ret in tqdm(p.imap_unordered(topk_seds_func, zip(it.count(), queries, lbs, ubs)), desc='queries', total=len(queries)):
                seds[i] = ret[0]
                idxs[i] = ret[1]
    return seds, idxs
