from . import config

import networkx as nx
import torch
import torch.nn
import torch_geometric as tg
import torch_geometric.data
import torch_geometric.utils
from tqdm.auto import tqdm

def batch_of_dataset(dataset):
    loader = tg.data.DataLoader(dataset, batch_size=len(dataset))
    for g, h, lb, ub in loader:
        g.batch = g.batch.to(config.device)
        h.batch = h.batch.to(config.device)
        return g, h, lb, ub

def to_pyged(g):
    return (torch.argmax(g.x, dim=1).tolist(), list(zip(*g.edge_index.tolist())))

def label_graphs(graphs, num_classes=None):
    tqdm.write('move labels to node attrs')
    ret = []
    if num_classes is None:
        num_classes = 1
        if(graphs[0].y is not None):
            num_classes = max([max(g['y']) for g in graphs]) + 1
    for g in tqdm(graphs, desc='graphs'):
        if g['y'] is None:
            g['y'] = torch.zeros(g.num_nodes, dtype=torch.long)
        g.x = torch.nn.functional.one_hot(g.y, num_classes).float()
        del g.y
        ret.append(g)
    return ret

def remove_extra_attrs(graphs):
    tqdm.write('remove extra attrs')
    ret = []
    for g in tqdm(graphs, desc='graphs'):
        for k in g.keys:
            if k not in ['x', 'edge_index']:
                g[k] = None
        ret.append(g)
    return ret

def similarity_of_sed(sed, gs):
    sed, gs = (x.to(config.device) for x in (sed, gs))
    return torch.exp(-sed/gs)

def similarity_of_ged(ged, gs, hs):
    ged, gs, hs = (x.to(config.device) for x in (ged, gs, hs))
    return torch.exp(-2*ged/(gs+hs))

def is_connected(g):
    return nx.is_connected(tg.utils.to_networkx(g, to_undirected=True))

def confusion_matrix(pred, gt, n_class):
    confusion_matrix = torch.zeros(pred.shape[0], n_class, n_class)
    for i in range(pred.shape[0]):
        for t, p in zip(gt[i].view(-1), pred[i].view(-1)):
            confusion_matrix[i, t.long(), p.long()] += 1
    return confusion_matrix

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def norm_sed_func(gx, hx):
    gx, hx = (x.to(config.device) for x in (gx, hx))
    return torch.norm(torch.nn.functional.relu(gx-hx), dim=-1)

def norm_ged_func(gx, hx):
    gx, hx = (x.to(config.device) for x in (gx, hx))
    return torch.norm(gx-hx, dim=-1)

def load_nbrs(inpath, total, num_classes=None):
    ret = []
    with open(inpath, 'r') as infile:
        for _ in tqdm(range(total), desc='neighborhoods', unit_scale=True, unit=''):
            y = torch.LongTensor([int(x) for x in infile.readline().split()])
            e0 = [int(x) for x in infile.readline().split()]
            e1 = [int(x) for x in infile.readline().split()]
            edge_index = torch.LongTensor([e0,e1])
            ret.append(tg.data.Data(edge_index=edge_index, y=y))
    return label_graphs(ret, num_classes)