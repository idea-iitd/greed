import IPython as ipy
import IPython.display
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric as tg
import torch_geometric.data
import torch_geometric.utils

from . import config, datasets, metrics

def plot_graph(g, shell=False, **kwargs):
    g = tg.data.Data(g.x, g.edge_index)
    nx_g = tg.utils.to_networkx(g.to(torch.device('cpu')), to_undirected=True)
    shells = []
    if shell:
        k = 0
        while True:
            shells.append(nx.descendants_at_distance(nx_g, 0, k))
            if not shells[-1]:
                break
            k += 1
    nx.draw(nx_g,
        pos=nx.shell_layout(nx_g, shells) if shell else None,
        node_color=torch.argmax(g.x, dim=1),
        vmin=0,
        vmax=g.x.shape[1]-1,
        cmap=plt.get_cmap('prism'),
#         cmap=plt.get_cmap('rainbow'),
        **kwargs)
    
def plot_pair(g, h, **kwargs):
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plot_graph(g, **kwargs)
    plt.subplot(122)
    plot_graph(h, **kwargs)

def plot_inner_dataset(dataset, pred=None, n_items=None, random=False, **kwargs):
    queries, targets, lb, ub = dataset
    if n_items is None:
        n_items = len(queries)
    if pred is not None:
        err = metrics.err(lb, ub, pred)
    for i in range(n_items):
        i = np.random.randint(len(queries)) if random else i
        q = queries[i]
        t = targets[i]
        plot_pair(q, t, **kwargs)
        if pred is None:
            plt.suptitle(f'#{i}    Q: ({q.num_nodes}, {q.num_edges})    T: ({t.num_nodes}, {t.num_edges})    LB: {lb[i]:.1f}    UB: {ub[i]:.1f}')
        else:
            plt.suptitle(f'#{i}    Q: ({q.num_nodes}, {q.num_edges})    T: ({t.num_nodes}, {t.num_edges})    LB: {lb[i]:.1f}    UB: {ub[i]:.1f}    PRED: {pred[i]:.1f}    ERR: {err[i]:.1f}')
        plt.show()
        ipy.display.display(ipy.display.Markdown('---'))
        
def plot_inner_dataset_plus(dataset, metaset, n_items=None, random=False, **kwargs):
    queries, targets, lb, ub = dataset
    node_masks, edge_masks = metaset
    ns_transform = datasets.make_target_transform(0)
    if n_items is None:
        n_items = len(queries)

    for i in range(n_items):
        i = np.random.randint(len(queries)) if random else i
        g = queries[i]
        h = targets[i]
        nm = node_masks[i]
        em = edge_masks[i]
        _, h_ns, _, _ = ns_transform(((g, h, lb[i], ub[i]), (nm, em)))
        
        plt.figure(figsize=(16,4))
        plt.subplot(131)
        plot_graph(g, **kwargs)
        plt.title(f'Q: ({g.num_nodes}, {g.num_edges})')
        plt.subplot(132)
        plot_graph(h, **kwargs)
        plt.title(f'T: ({h.num_nodes}, {h.num_edges})')
        plt.subplot(133)
        plot_graph(h_ns, **kwargs)
        plt.title(f'NS: ({h_ns.num_nodes}, {h_ns.num_edges})')
        
        plt.suptitle(f'#{i}    LB: {lb[i]:.1f}    UB: {ub[i]:.1f}')
#         plt.savefig(f'{i}.png')
#         plt.close()
        plt.show()
        ipy.display.display(ipy.display.Markdown('---'))
        
def plot_outer_dataset(dataset, pred=None, n_items=None, random=False, **kwargs):
    queries, targets, lb, ub = dataset
    if n_items is None:
        n_items = len(dataset)
    if pred is not None:
        err = metrics.err(lb, ub, pred)
    for i in range(n_items):
        i = np.random.randint(len(queries)*len(targets)) if random else i
        qi = i//len(targets)
        ti = i%len(targets)
        q = queries[i]
        t = targets[i]
        plot_pair(queries[qi], targets[ti])
        if pred is None:
            plt.suptitle(f'#{i}    Q: ({q.num_nodes}, {q.num_edges})    T: ({t.num_nodes}, {t.num_nodes})    LB: {lb[qi,ti]}    UB: {ub[qi,ti]:.1f}')
        else:
            plt.suptitle(f'#{i}    Q: ({q.num_nodes}, {q.num_edges})    T: ({t.num_nodes}, {t.num_nodes})    LB: {lb[qi,ti]}    UB: {ub[qi,ti]:.1f}    PRED: {pred[qi,ti]:.1f}    ERR: {err[qi,ti]:.1f}')
        plt.show()
        ipy.display.display(ipy.display.Markdown('---'))

def plot_aligned(g, h, pos=None, **kwargs):
    g = tg.data.Data(g.x, g.edge_index)
    h = tg.data.Data(h.x, h.edge_index)
    nx_g = tg.utils.to_networkx(g.to(torch.device('cpu')), to_undirected=True)
    nx_h = tg.utils.to_networkx(h.to(torch.device('cpu')), to_undirected=True)
    pos_g = pos if pos is not None else nx.spring_layout(nx_g)
    pos_h = pos_g
    plt.subplot(121)
    nx.draw(nx_g,
        pos=pos_g,
        node_color=torch.argmax(g.x, dim=1),
        vmin=0,
        vmax=g.x.shape[1]-1,
        cmap=plt.get_cmap('rainbow'),
        **kwargs)
    plt.subplot(122)
    nx.draw(nx_h,
        pos=pos_h,
        node_color=torch.argmax(h.x, dim=1),
        vmin=0,
        vmax=h.x.shape[1]-1,
        cmap=plt.get_cmap('rainbow'),
        **kwargs)
        
def plot_dist(x, bin_size=1, xmin=None, xmax=None):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    else:
        x = torch.tensor(x)
    if xmin is None:
        xmin = torch.min(x)
    if xmax is None:
        xmax = torch.max(x)
    xmin = int(xmin)
    xmax = int(xmax)
    plt.hist(x.numpy().flatten(), bins=range(xmin, xmax+1, bin_size))

def plot_node_dist(graphs, bin_size=1):
    x = [g.num_nodes for g in graphs]
    plot_dist(x, bin_size=bin_size)
    plt.xlabel('#nodes')
    plt.ylabel('#graphs')

def plot_edge_dist(graphs, bin_size=2):
    x = [g.num_edges for g in graphs]
    plot_dist(x, bin_size=bin_size)
    plt.xlabel('#edges')
    plt.ylabel('#graphs')

def plot_size_dist(graphs, node_bin_size=1, edge_bin_size=2):
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plot_node_dist(graphs, node_bin_size)
    plt.title('Node Distribution')
    plt.subplot(122)
    plot_edge_dist(graphs, edge_bin_size)
    plt.title('Edge Distribution')

def plot_dataset_dist(dataset, node_bin_size=1, edge_bin_size=2):
    queries, targets, lb, ub = dataset
    plot_size_dist(queries, node_bin_size, edge_bin_size)
    plt.suptitle(f'Queries (total: {len(queries)})')
    plt.show()
    plot_size_dist(targets, node_bin_size, edge_bin_size)
    plt.suptitle(f'Targets (total: {len(targets)})')
    plt.show()
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plot_dist(torch.flatten(lb))
    plt.xlabel('lb')
    plt.ylabel('#pairs')
    plt.title('LB')
    plt.subplot(122)
    plot_dist(torch.flatten(ub))
    plt.title('UB')
    plt.xlabel('ub')
    plt.ylabel('#pairs')
    plt.show()

def plot_corr(x, y, xmax=None, bin_size=1, show_std=False, *args, **kwargs):
    x = x.cpu()
    if xmax is None:
        xmax = torch.max(x)
    y = y[x<xmax+1]
    x = x[x<xmax+1]
    x = x//bin_size*bin_size
    xx = torch.unique(x)
    mean = torch.tensor([torch.mean(y[x == i]) for i in xx])
    if show_std:
        std = torch.tensor([torch.std(y[x == i]) for i in xx])
        std[torch.isnan(std)] = 0
    plt.plot(xx, mean, *args, **kwargs)
    if show_std:
        plt.fill_between(xx, torch.max(torch.zeros_like(mean), mean-std), mean+std, alpha=0.2)
        
def plot_corr_2d(x1, x2, y, xmax1=None, xmax2=None, bin_size1=1, bin_size2=1, **kwargs):
    x1 = x1.cpu()
    x2 = x2.cpu()
    if xmax1 is None:
        xmax1 = torch.max(x1)
    if xmax2 is None:
        xmax2 = torch.max(x2)
    y = y[x1<xmax1+1]
    x2 = x2[x1<xmax1+1]
    x1 = x1[x1<xmax1+1]
    y = y[x2<xmax2+1]
    x1 = x1[x2<xmax2+1]
    x2 = x2[x2<xmax2+1]
    x1 = x1//bin_size1*bin_size1
    x2 = x2//bin_size2*bin_size2
    xx1 = torch.unique(x1)
    xx2 = torch.unique(x2)
    mean = torch.tensor([[torch.mean((y[x2 == j])[x1[x2 == j] == i]) for j in xx2] for i in xx1])
    _x1,_x2,_y = [],[],[]
    for i in range(len(xx1)):
        for j in range(len(xx2)):
            if(not torch.isnan(mean[i][j])):
                _x1.append(xx1[i])
                _x2.append(xx2[j])
                _y.append(mean[i][j])
    plt.scatter(_x1, _x2, c=_y, **kwargs)

def plot_summary(x, Y, lim=True, show_std=True, *args, **kwargs):
    with torch.no_grad():
        x = x.cpu()
        Y = Y.to(config.device)
        isnan = torch.isnan(Y)
        Y[isnan] = 0
        mean = torch.sum(Y, dim=-1)/torch.sum(~isnan, dim=-1)
        if show_std:
            std = torch.sqrt(torch.sum(((Y-mean[:,None])*(~isnan))**2, dim=-1) / torch.sum(~isnan, dim=-1))
            std = std.cpu()
        mean = mean.cpu()
        plt.plot(x, mean, *args, **kwargs)
        if show_std:
            if lim:
                plt.fill_between(x, torch.max(torch.zeros_like(mean), mean-std),
                                 torch.min(torch.ones_like(mean), mean+std), alpha=0.2)
            else:
                plt.fill_between(x, torch.max(torch.zeros_like(mean), mean-std), mean+std, alpha=0.2)
        if lim:
            plt.ylim((0,1.05))
        Y[isnan] = float('nan')

def plot_individual(x, Y, lim=True):
    with torch.no_grad():
        x = x.cpu()
        Y = Y.cpu()
        for y in Y.t():
            plt.plot(x, y, alpha=100/Y.shape[1])
        if lim:
            plt.ylim((0,1.05))