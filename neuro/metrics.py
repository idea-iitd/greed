import sklearn as sk
import sklearn.metrics
import torch
import torch_geometric as tg
import torch_geometric.data
from tqdm.auto import tqdm

from . import config, utils
import sys
sys.path.insert(0, '../..')
sys.path.insert(0, '../../pyged/lib')
import pyged

def err(lb, ub, pred):
    lb, ub, pred = (x.to(config.device) for x in (lb, ub, pred))
    return torch.nn.functional.relu(lb-pred) + torch.nn.functional.relu(pred-ub)

def mse(lb, ub, pred, w=1, **kwargs):
    if isinstance(w, torch.Tensor):
        w = w.to(config.device)
    se = err(lb, ub, pred)**2
    return torch.sum(w*se, **kwargs) / torch.sum(w*torch.ones_like(se), **kwargs)

def weighted_mse(lb, ub, pred, **kwargs):
    lb, ub = (x.to(config.device) for x in (lb, ub))
    return mse(lb, ub, pred, w=1/((lb+ub)/2+1), **kwargs)

def rmse(lb, ub, pred, **kwargs):
    return torch.sqrt(mse(lb, ub, pred, **kwargs))

def weighted_rmse(lb, ub, pred, **kwargs):
    return torch.sqrt(weighted_mse(lb, ub, pred, **kwargs))

def mae(lb, ub, pred, w=1, **kwargs):
    if isinstance(w, torch.Tensor):
        w = w.to(config.device)
    ae = err(lb, ub, pred)
    return torch.sum(w*ae, **kwargs) / torch.sum(w*torch.ones_like(ae), **kwargs)

def weighted_mae(lb, ub, pred, **kwargs):
    lb, ub = (x.to(config.device) for x in (lb, ub))
    return mae(lb, ub, pred, w=1/((lb+ub)/2+1), **kwargs)

def r2(lb, ub, pred, w=1, **kwargs):
    lb, ub = (x.to(config.device) for x in (lb, ub))
    if 'dim' in kwargs:
        dmse = mse(lb, ub, torch.mean((lb+ub)/2, keepdim=True, **kwargs))
    else:
        dmse = mse(lb, ub, torch.mean((lb+ub)/2, **kwargs))
    return 1 - mse(lb, ub, pred, w=w, **kwargs) / dmse

def weighted_r2(lb, ub, pred, **kwargs):
    lb, ub = (x.to(config.device) for x in (lb, ub))
    return r2(lb, ub, pred, w=1/((lb+ub)/2+1), **kwargs)

def rmse_at_k(k, lb, ub, pred):
    with torch.no_grad():
        k, lb, ub, pred = (x.to(config.device) for x in (k, lb, ub, pred))
        _, pred_sort_idx = torch.sort(pred, dim=-1)
        lb_ranked = torch.gather(lb, -1, pred_sort_idx)
        ub_sorted, _ = torch.sort(ub, dim=-1)
        ar = torch.arange(lb.shape[1]).to(config.device)
        mse = torch.sum((ar[None,None,:] < k[:,None,None]) * \
                        torch.nn.functional.relu(lb_ranked - ub_sorted)**2, dim=-1) / k[:,None].float()
        return torch.sqrt(mse)

def error_at_k(k, lb, ub, pred):
    with torch.no_grad():
        k, lb, ub, pred = (x.to(config.device) for x in (k, lb, ub, pred))
        _, pred_sort_idx = torch.sort(pred, dim=-1)
        lb_ranked = torch.gather(lb, -1, pred_sort_idx)
        ub_sorted, _ = torch.sort(ub, dim=-1)
        ar = torch.arange(lb.shape[1]).to(config.device)
        return torch.nn.functional.relu(torch.sum((ar[None,None,:] < k[:,None,None]) * \
                                                  (lb_ranked - ub_sorted), dim=-1)) / k[:,None].float()
    
def precision_at_k(k, lb, ub, pred):
    with torch.no_grad():
        k, lb, ub, pred = (x.to(config.device) for x in (k, lb, ub, pred))
        _, pred_sort_idx = torch.sort(pred, dim=-1)
        lb_ranked = torch.gather(lb, -1, pred_sort_idx)
        ub_sorted, _ = torch.sort(ub, dim=-1)
        ub_sorted = torch.transpose(ub_sorted, -2, -1)
        ar = torch.arange(lb.shape[-1]).to(config.device)
        return torch.sum((ar[None,None,:] < k[:,None,None]) & \
                         (lb_ranked[None,:,:] <= ub_sorted[k-1,:,None]), dim=-1) / k[:,None].float()
    
def dcg_at_k(k, sed):
    rel = 1/(sed+1)
    ar = torch.arange(sed.shape[-1]).to(config.device)
    gain = rel/torch.log2(ar+2)
    return torch.sum((ar[None,None,:] < k[:,None,None]) * gain, dim=-1)
    
def ndcg_at_k(k, lb, ub, pred):
    with torch.no_grad():
        k, lb, ub, pred = (x.to(config.device) for x in (k, lb, ub, pred))
        sed = (lb+ub)/2
        _, pred_sort_idx = torch.sort(pred, dim=-1)
        sed_ranked = torch.gather(sed, -1, pred_sort_idx)
        sed_sorted, _ = torch.sort(sed, dim=-1)
        return dcg_at_k(k, sed_ranked) / dcg_at_k(k, sed_sorted)
    
def kendalls_tau_at_k(k, lb, ub, pred):
    with torch.no_grad():
        k, lb, ub, pred = (x.to(config.device) for x in (k, lb, ub, pred))
        sed = (lb+ub)/2
        _, sed_sort_idx = torch.sort(sed, dim=-1)
        lb_ranked = torch.gather(lb, -1, sed_sort_idx)
        ub_ranked = torch.gather(ub, -1, sed_sort_idx)
        pred_ranked = torch.gather(pred, -1, sed_sort_idx)
        dsc = ((pred[:,:,None] < pred[:,None,:]) & (lb[:,:,None] > ub[:,None,:]))
        ret = torch.empty(k.shape[-1], lb.shape[-2]).to(config.device)
        for i in tqdm(range(len(k)), desc='k', disable=True):
            d = torch.sum(torch.sum(dsc[:,:k[i],:k[i]], dim=-1), dim=-1)
            ret[i,:] = 1 - 4*d/k[i]/(k[i]-1)
        return ret

def accuracy_at_range(r, lb, ub, pred):
    with torch.no_grad():
        r, lb, ub, pred = (x.to(config.device) for x in (r, lb, ub, pred))
        lb_mask = lb[None,:,:] <= r[:,None,None]
        ub_mask = ub[None,:,:] > r[:,None,None]
        pred_mask = pred[None,:,:] <= r[:,None,None]
    return torch.sum((pred_mask & lb_mask) | (~pred_mask & ub_mask), dim=-1) / float(lb.shape[1])

def precision_at_range(r, lb, ub, pred):
    with torch.no_grad():
        r, lb, ub, pred = (x.to(config.device) for x in (r, lb, ub, pred))
        lb_mask = lb[None,:,:] <= r[:,None,None]
        pred_mask = pred[None,:,:] <= r[:,None,None]
        return torch.sum(pred_mask & lb_mask, dim=-1) / torch.sum(pred_mask, dim=-1).float()

def recall_at_range(r, lb, ub, pred):
    with torch.no_grad():
        r, lb, ub, pred = (x.to(config.device) for x in (r, lb, ub, pred))
        ub_mask = ub[None,:,:] < r[:,None,None]
        pred_mask = pred[None,:,:] <= r[:,None,None]
        return torch.sum(pred_mask & ub_mask, dim=-1) / torch.sum(ub_mask, dim=-1).float()

def f1_at_range(r, lb, ub, pred):
    prec = precision_at_range(r, lb, ub, pred)
    rec = recall_at_range(r, lb, ub, pred)
    return 2*prec*rec/(prec+rec)

def auroc_at_range(r, lb, ub, pred):
    with torch.no_grad():
        true = (lb+ub)/2
        ret = torch.empty(r.shape[0], lb.shape[0]).to(config.device)
        for i in range(r.shape[0]):
            y_true = true < r[i]
            y_score = -pred
            y_true = y_true.cpu()
            y_score = y_score.cpu()
            for j in range(lb.shape[0]):
                if torch.all(y_true[j] == y_true[j,0]):
                    ret[i,j] = float('nan')
                else:
                    ret[i,j] = sk.metrics.roc_auc_score(y_true[j], y_score[j])
        return ret
    
def topk_set_aux(k, seds, idxs):
    ret = idxs[:k]
    for i in range(k, len(seds)):
        if seds[i] > seds[k-1]:
            break
        ret.append(idxs[i])
    return ret

def topk_set(k, seds, idxs):
    ret = [None] * len(seds)
    for i in range(len(seds)):
        ret[i] = topk_set_aux(k, seds[i], idxs[i])
    return ret

def exact_precision_at_k(ks, seds, idxs, pred):
    pred = pred.cpu()
    _, pred_sort_idx = torch.sort(pred, dim=-1)
    ret = torch.empty(len(ks), len(pred))
    for ki, k in enumerate(ks):
        topk_pred = pred_sort_idx[...,:k]
        topk_true = topk_set(k, seds, idxs)
        for i in range(len(pred)):
            ret[ki,i] = len(set(topk_pred[i].tolist()) & set(topk_true[i]))/k
    return ret