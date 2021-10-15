from . import config, utils

import IPython as ipy
import matplotlib.pyplot as plt
import torch
import torch_geometric as tg
import torch_geometric.data
from tqdm.auto import tqdm

import copy
import itertools as it
import os
import time
import numpy as np

def train_epoch(model, opt, loader, max_grad_norm=config.max_grad_norm):
    model.train()
    losses = []
    for g, h, lb, ub in tqdm(loader, desc='batches', leave=False):
        g = g.to(config.device)
        h = h.to(config.device)
        lb = lb.to(config.device)
        ub = ub.to(config.device)
        pred = model(g, h)
        loss = model.criterion(lb, ub, pred)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
    return losses

def train_loop(model, opt, loader, n_epochs=None, max_grad_norm=config.max_grad_norm):
    losses = []
    ectr = 0
    etime = 0
    while True:
        tic = time.time()
        new_losses = train_epoch(model, opt, loader)
        toc = time.time()

        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.plot(new_losses)
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.title('Last Epoch')
        losses += new_losses
        plt.subplot(122)
        plt.plot(losses)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('All Epochs')
        ipy.display.clear_output(wait=True)
        plt.show()

        ectr += 1
        if ectr == n_epochs:
            break
        etime += toc-tic
        print(f'epoch: {ectr}\ttotal time: {etime:.3f} s\ttime per epoch: {etime/ectr:.3f} s')

def train_epoch_val(model, opt, loader, val_loader, lrs=None, dump_path=None, max_grad_norm=config.max_grad_norm):
    model.train()
    losses = []
    val_losses = []
#     for batch, val_batch in zip(tqdm(loader, desc='batches', leave=False), it.cycle(val_loader)):#

    for batch, val_batch in zip(loader, it.cycle(val_loader)):

        val_g, val_h, val_lb, val_ub = (x.to(config.device) for x in val_batch)
        with torch.no_grad():
            val_pred = model(val_g, val_h)
            val_loss = model.criterion(val_lb, val_ub, val_pred)
        val_losses.append(val_loss.item())
            
        g, h, lb, ub = (x.to(config.device) for x in batch)
        pred = model(g, h)
        loss = model.criterion(lb, ub, pred)
        losses.append(loss.item())
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        if lrs is not None:
            lrs.step()
        
    return losses, val_losses

def train_loop_val(model, opt, loader, val_loader, n_epochs=None, max_grad_norm=config.max_grad_norm):
    losses = []
    val_losses = []
    ectr = 0
    etime = 0
    while True:
        tic = time.time()
        new_losses, new_val_losses = train_epoch_val(model, opt, loader, val_loader)
        toc = time.time()

        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.plot(new_losses, label='train')
        plt.plot(new_val_losses, label='val')
        plt.legend()
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.title('Last Epoch')
        losses += new_losses
        val_losses += new_val_losses
        
        plt.subplot(122)
        plt.plot(losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('All Epochs')
        ipy.display.clear_output(wait=True)
        plt.show()

        ectr += 1
        if ectr == n_epochs:
            break
        etime += toc-tic
        print(f'epoch: {ectr}\ttotal time: {etime:.3f} s\ttime per epoch: {etime/ectr:.3f} s')
        
def train_full(model, loader, val_loader, lr, weight_decay, cycle_patience, step_size_up, step_size_down, dump_path=None, lrs=None, max_grad_norm=config.max_grad_norm):
    
    tqdm.write(f'dump path: {dump_path}')
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lrs = lrs if lrs is not None else torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0, max_lr=lr, step_size_up=step_size_up, step_size_down=step_size_down, cycle_momentum=False)

    all_losses = []
    all_val_losses = []
    ectr = 0
    best_val_loss = float('inf')
    bctr = 0
    tic = time.time()
    while True:
        ectr += 1
        
        losses = []
        val_losses = []
        for batch, val_batch in zip(loader, it.cycle(val_loader)):
            
            val_g, val_h, val_lb, val_ub = (x.to(config.device) for x in val_batch)
            with torch.no_grad():
                val_pred = model(val_g, val_h)
                val_loss = model.criterion(val_lb, val_ub, val_pred)
            val_losses.append(val_loss.item())
            if val_loss >= best_val_loss:
                bctr += 1
                if bctr > cycle_patience * (step_size_up+step_size_down):
                    model.load_state_dict(best_model)
                    return all_losses, all_val_losses
            else:
                bctr = 0
                best_model = copy.deepcopy(model.state_dict())
                if dump_path is not None:
                    torch.save(best_model, os.path.join(dump_path, 'best_model.pt'))
                best_val_loss = val_loss

            g, h, lb, ub = (x.to(config.device) for x in batch)
            pred = model(g, h)
            loss = model.criterion(lb, ub, pred)
            losses.append(loss.item())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            lrs.step()

        all_losses += losses
        all_val_losses += val_losses
        
        plt.figure()
        plt.plot(all_losses, label='train')
        plt.plot(all_val_losses, label='val')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('loss')
        
        if val_losses[0] <= 1:
            plt.ylim((0,1))
        elif val_losses[0] <= 5:
            plt.ylim((0,5))
        elif val_losses[0] <= 10:
            plt.ylim((0,10))
        elif val_losses[0] <= 100:
            plt.ylim((0,100))
            
        plt.title(f'epoch:{ectr} | time:{int(time.time()-tic)}s | val:{val_loss:.3} | best:{best_val_loss:.3} | patience:{bctr//(step_size_up+step_size_down)}/{cycle_patience}')
        if utils.is_notebook() and not dump_path:
            ipy.display.clear_output(wait=True)
            plt.show()
        else:
            assert dump_path is not None
            plt.savefig(os.path.join(dump_path, 'loss_curve.png'))
            plt.close()
        
        if dump_path is not None:
            torch.save(all_losses, os.path.join(dump_path, 'losses.pt'))
            torch.save(all_val_losses, os.path.join(dump_path, 'val_losses.pt'))
            torch.save(lrs, os.path.join(dump_path, 'lrs.pt'))
            torch.save(model.state_dict(), os.path.join(dump_path, 'model.pt'))
            with open(os.path.join(dump_path, 'losses.txt'),'a') as file:
                file.write(f'train:{np.average(losses)}\t val:{np.average(val_losses)}\n')