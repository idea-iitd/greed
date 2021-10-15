from . import config

import torch
import torch_geometric as tg
from tqdm.auto import tqdm

import itertools as it

class EmbedModel(torch.nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, conv='gin', pool='add'):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.pre = torch.nn.Linear(self.input_dim, self.hidden_dim)
        
        if conv == 'gin':
            make_conv = lambda:\
                tg.nn.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                ))
        elif conv == 'gcn':
            make_conv = lambda:\
                tg.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv == 'sage':
            make_conv = lambda:\
                tg.nn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv == 'gat':
            make_conv = lambda:\
                tg.nn.GATConv(self.hidden_dim, self.hidden_dim)
        else:
            assert False
            
        self.convs = torch.nn.ModuleList()
        for l in range(self.n_layers):
            self.convs.append(make_conv())
        
        self.post = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim*(self.n_layers+1), self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        if pool == 'add':
            self.pool = tg.nn.global_add_pool
        elif pool == 'mean':
            self.pool = tg.nn.global_mean_pool
        elif pool == 'max':
            self.pool = tg.nn.global_max_pool
        elif pool == 'sort':
            self.pool = tg.nn.global_sort_pool
        elif pool == 'att':
            self.pool = tg.nn.GlobalAttention(torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim*(self.n_layers+1), self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, 1)
            ))
        elif pool == 'set':
            self.pool = tg.nn.Set2Set(self.hidden_dim*(self.n_layers+1),1)
        self.pool_str = pool
        
    def forward(self, g):
        x = g.x
        edge_index = g.edge_index

        x = self.pre(x)
        emb = x
        xres = x
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i&1:
                x += xres
                xres = x
            x = torch.nn.functional.relu(x)
            emb = torch.cat((emb, x), dim=1)
        
        x = emb
        if self.pool_str == 'sort':
            x = self.pool(x, g.batch, k=1)
        else:
            x = self.pool(x, g.batch)
        
        x = self.post(x)
        return x

class SiameseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_model = None
        self.weighted = False
        
    def forward_emb(self, gx, hx):
        raise NotImplementedError

    def forward(self, g, h):
        if self.weighted:
            self.gs = torch.tensor([x.num_nodes for x in g.to_data_list()], device=config.device)
            self.hs = torch.tensor([x.num_nodes for x in h.to_data_list()], device=config.device)
        gx = self.embed_model(g)
        hx = self.embed_model(h)
        return self.forward_emb(gx, hx)
    
    def predict_inner(self, queries, targets, batch_size=None):
        self = self.to(config.device)
        if batch_size is None or len(queries) <= batch_size:
            tqdm.write(f'direct predict inner dataset')
            g = tg.data.Batch.from_data_list(queries).to(config.device)
            h = tg.data.Batch.from_data_list(targets).to(config.device)
            with torch.no_grad():
                return self.forward(g, h)
        else:
            tqdm.write(f'batch predict inner dataset')
            tqdm.write(f'config.n_workers: {config.n_workers}')
            loader = tg.data.DataLoader(list(zip(queries, targets)), batch_size, num_workers=config.n_workers)
            ret = torch.empty(len(queries), device=config.device)
            for i, (g, h) in enumerate(tqdm(loader, 'batches')):
                g = g.to(config.device)
                h = h.to(config.device)
                with torch.no_grad():
                    ret[i*batch_size:(i+1)*batch_size] = self.forward(g, h)
            return ret
    
    def predict_outer(self, queries, targets, batch_size=None):
        self = self.to(config.device)
        if batch_size is None or len(queries)*len(targets) <= batch_size:
            tqdm.write(f'direct predict outer dataset')
            g = tg.data.Batch.from_data_list(queries).to(config.device)
            h = tg.data.Batch.from_data_list(targets).to(config.device)
            gx = self.embed_model(g)
            hx = self.embed_model(h)
            with torch.no_grad():
                return self.forward_emb(gx[:,None,:], hx)
        else:
            tqdm.write(f'batch predict outer dataset')
            tqdm.write(f'config.n_workers: {config.n_workers}')
            g = tg.data.Batch.from_data_list(queries).to(config.device)
            gx = self.embed_model(g)
            loader = tg.data.DataLoader(targets, batch_size//len(queries), num_workers=config.n_workers)
            ret = torch.empty(len(queries), len(targets), device=config.device)
            for i, h in enumerate(tqdm(loader, 'batches')):
                h = h.to(config.device)
                hx = self.embed_model(h)
                with torch.no_grad():
                    ret[:,i*loader.batch_size:(i+1)*loader.batch_size] = self.forward_emb(gx[:,None,:], hx)
            return ret

    def criterion(self, lb, ub, pred):
        loss = torch.nn.functional.relu(lb-pred)**2 + torch.nn.functional.relu(pred-ub)**2
        if self.weighted:
            loss /= ((self.gs+self.hs)/2)**2
        loss = torch.mean(loss)
        return loss
    
class NeuralSiameseModel(SiameseModel):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embed_model = EmbedModel(n_layers, input_dim, hidden_dim, output_dim)
        self.mlp_model = torch.nn.Sequential(
            torch.nn.Linear(2*output_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, 1)
        )
    
    def forward_emb(self, gx, hx):
        if gx.dim() == hx.dim():
            return self.mlp_model(torch.cat((gx, hx), dim=-1)).view(-1)
        else:
            gx = gx[:,0,:]
            return self.mlp_model(torch.cat((torch.repeat_interleave(gx,hx.shape[0],dim=0), torch.tile(hx,[gx.shape[0],1])), dim=-1)).view(gx.shape[0],hx.shape[0])

class NormGEDModel(SiameseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_model = EmbedModel(*args, **kwargs)
        
    def forward_emb(self, gx, hx):
        return torch.norm(gx-hx, dim=-1)

class NormSEDModel(SiameseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_model = EmbedModel(*args, **kwargs)
    
    def forward_emb(self, gx, hx):
        return torch.norm(torch.nn.functional.relu(gx-hx), dim=-1)
    
class DualSiameseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_model_g = None
        self.embed_model_h = None
        self.weighted = False
        
    def forward_emb(self, gx, hx):
        raise NotImplementedError

    def forward(self, g, h):
        gx = self.embed_model_g(g)
        hx = self.embed_model_h(h)
        return self.forward_emb(gx, hx)
    
    def predict_inner(self, queries, targets, batch_size=None):
        self = self.to(config.device)
        if batch_size is None or len(queries) <= batch_size:
            tqdm.write(f'direct predict inner dataset')
            g = tg.data.Batch.from_data_list(queries).to(config.device)
            h = tg.data.Batch.from_data_list(targets).to(config.device)
            with torch.no_grad():
                return self.forward(g, h)
        else:
            tqdm.write(f'batch predict inner dataset')
            tqdm.write(f'config.n_workers: {config.n_workers}')
            loader = tg.data.DataLoader(list(zip(queries, targets)), batch_size, num_workers=config.n_workers)
            ret = torch.empty(len(queries), device=config.device)
            for i, (g, h) in enumerate(tqdm(loader, 'batches')):
                g = g.to(config.device)
                h = h.to(config.device)
                with torch.no_grad():
                    ret[i*batch_size:(i+1)*batch_size] = self.forward(g, h)
            return ret
    
    def predict_outer(self, queries, targets, batch_size=None):
        self = self.to(config.device)
        if batch_size is None or len(queries)*len(targets) <= batch_size:
            tqdm.write(f'direct predict outer dataset')
            g = tg.data.Batch.from_data_list(queries).to(config.device)
            h = tg.data.Batch.from_data_list(targets).to(config.device)
            gx = self.embed_model_g(g)
            hx = self.embed_model_h(h)
            with torch.no_grad():
                return self.forward_emb(gx[:,None,:], hx)
        else:
            tqdm.write(f'batch predict outer dataset')
            tqdm.write(f'config.n_workers: {config.n_workers}')
            g = tg.data.Batch.from_data_list(queries).to(config.device)
            gx = self.embed_model_g(g)
            loader = tg.data.DataLoader(targets, batch_size//len(queries), num_workers=config.n_workers)
            ret = torch.empty(len(queries), len(targets), device=config.device)
            for i, h in enumerate(tqdm(loader, 'batches')):
                h = h.to(config.device)
                hx = self.embed_model_h(h)
                with torch.no_grad():
                    ret[:,i*loader.batch_size:(i+1)*loader.batch_size] = self.forward_emb(gx[:,None,:], hx)
            return ret

    def criterion(self, lb, ub, pred):
        loss = torch.nn.functional.relu(lb-pred)**2 + torch.nn.functional.relu(pred-ub)**2
        if self.weighted:
            loss /= ((lb+ub)/2+1)**2
        loss = torch.mean(loss)
        return loss
    
class DualNeuralSiameseModel(DualSiameseModel):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embed_model_g = EmbedModel(n_layers, input_dim, hidden_dim, output_dim)
        self.embed_model_h = EmbedModel(n_layers, input_dim, hidden_dim, output_dim)
        self.mlp_model = torch.nn.Sequential(
            torch.nn.Linear(2*output_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, 1)
        )
    
    def forward_emb(self, gx, hx):
        return self.mlp_model(torch.cat((gx, hx), dim=-1)).view(-1)

class DualNormGEDModel(DualSiameseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_model_g = EmbedModel(*args, **kwargs)
        self.embed_model_h = EmbedModel(*args, **kwargs)
        
    def forward_emb(self, gx, hx):
        return torch.norm(gx-hx, dim=-1)

class DualNormSEDModel(DualSiameseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_model_g = EmbedModel(*args, **kwargs)
        self.embed_model_h = EmbedModel(*args, **kwargs)
    
    def forward_emb(self, gx, hx):
        return torch.norm(torch.nn.functional.relu(gx-hx), dim=-1)