from . import config

import torch
import torch.nn
import torch_geometric as tg
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils
from tqdm.auto import tqdm

def preprocess(graphs, max_degree):
    ret = []
    for g in tqdm(graphs, 'graphs'):
        deg = tg.utils.degree(g.edge_index[0], g.num_nodes).long()
        deg[deg > max_degree] = max_degree
        deg = torch.nn.functional.one_hot(deg, num_classes=max_degree+1).float()
        g.x = torch.cat((g.x, deg), dim=-1)
        ret.append(g)
    return ret

def sim_to_ged(sim, gs, hs):
    sim, gs, hs = (x.to(config.device) for x in (sim, gs, hs))
    return - (gs + hs)/2 * torch.log(sim)

def sim_to_sed(sim, gs, hs=None):
    sim, gs = (x.to(config.device) for x in (sim, gs))
    return - gs * torch.log(sim)

class DistancePredictor:
    def __init__(self, model, sim_to_dist=sim_to_sed, batch_size=None):
        self.model = model
        self.batch_size = batch_size
        self.sim_to_dist = sim_to_dist
        
    def __call__(self, query, targets):
        if not isinstance(targets, list):
            targets = [targets]
        queries = [query] * len(targets)
        self.model = self.model.to(config.device)
        if self.batch_size is None or len(queries) <= self.batch_size:
            with torch.no_grad():
                self.model.eval()
                g = tg.data.Batch.from_data_list(queries).to(config.device)
                h = tg.data.Batch.from_data_list(targets).to(config.device)
                sim = self.model(dict(g1=g, g2=h))
                self.model.train()
        else:
            loader = tg.data.DataLoader(list(zip(queries, targets)), self.batch_size)
            with torch.no_grad():
                self.model.eval()
                sim = torch.empty(len(queries), device=config.device)
                for i, (g, h) in enumerate(tqdm(loader, 'batches')):
                    g = g.to(config.device)
                    h = h.to(config.device)
                    sim[i*self.batch_size:(i+1)*self.batch_size] = self.model(dict(g1=g, g2=h))
                self.model.train()
        gs = torch.tensor([x.num_nodes for x in queries]).to(config.device)
        hs = torch.tensor([x.num_nodes for x in targets]).to(config.device)
        dist = self.sim_to_dist(sim, gs, hs)
        return dist