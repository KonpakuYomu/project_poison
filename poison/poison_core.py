import ssl
import torch
import lpips
import torch.nn as nn
from tqdm import tqdm
ssl._create_default_https_context = ssl._create_unverified_context


def glaze(x, x_trans, model, p=0.05, alpha=30, iters=500, lr=1e-2):
    delta = (torch.rand(*x.shape) * 2 * p - p).to(x.device)
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([delta], lr=lr)
    loss_fn_alex = lpips.LPIPS(net='vgg').to(x.device)

    for i in pbar:
        delta.requires_grad_(True)
        x_adv = x + delta
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        x_emb = model(x_adv).latent_dist.sample()
        x_trans_emb = model(x_trans).latent_dist.sample()
        
        optimizer.zero_grad()
        d = loss_fn_alex(x, x_adv)
        sim_loss = alpha * max(d-p, 0)
        loss = criterion(x_emb, x_trans_emb) + sim_loss
        
        loss.backward()
        optimizer.step()
        pbar.set_description(f"[Running glaze]: Loss {loss.item():.5f} | sim loss {alpha * max(d.item()-p, 0):.5f} | dist {d.item():.5f}")
        
    x_adv = x + delta
    x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
    return x_adv


def poi(x, x_trans, model, p1=0.1, p2=0.0025, alpha=10, iters=300, lr=1e-2):
    delta1 = (torch.randn(*x.shape) * 2 * p1 - p1).to(x.device)
    with torch.no_grad():
        x_trans_emb = model.encode(x_trans).latent_dist.sample()
    delta2 = (torch.randn(*x_trans_emb.shape) * 2 * p2 - p2).to(x_trans_emb.device)
    
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([delta2], lr=lr)
    
    for i in pbar:
        optimizer.zero_grad()
        delta1.requires_grad_(True)
        delta2.requires_grad_(True)
        
        x_adv = x + delta1
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        x_emb = model.encode(x_adv).latent_dist.sample()
        target = x_trans_emb + delta2
        target_dec = model.decode(target).sample

        shift_loss = criterion(target, x_emb)
        target_loss = max(criterion(target_dec, x_trans) - p2, 0)
        loss = shift_loss + alpha * target_loss
        loss.backward()
        
        delta1 = delta1 - (2 / 255) * delta1.grad.sign()
        delta1 = torch.clamp(delta1, min=-p1, max=+p1).detach_()
        optimizer.step()
        pbar.set_description(f"[Running poi]: Loss {loss.item():.5f} | shift loss {shift_loss.item():.5f} | target loss {target_loss:.5f}")

    with torch.no_grad():
        x_adv = x + delta1
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        target = x_trans_emb + delta2
        target_dec = model.decode(target).sample
    return x_adv, target_dec

def poi_decode_without_target_perturbation(x, x_trans, model, p1=0.1, iters=300):
    delta1 = (torch.randn(*x.shape) * 2 * p1 - p1).to(x.device)
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    
    for i in pbar:
        delta1.requires_grad_(True)
        x_adv = x + delta1
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        x_emb = model.encode(x_adv).latent_dist.sample()
        x_dec = model.decode(x_emb).sample
        loss = criterion(x_dec, x_trans)
        loss.backward()
        
        delta1 = delta1 - (2 / 255) * delta1.grad.sign()
        delta1 = torch.clamp(delta1, min=-p1, max=p1).detach_()
        pbar.set_description(f"[Running poi]: Loss {loss.item():.5f}")

    with torch.no_grad():
        x_adv = x + delta1
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
    return x_adv
    
def poi_decode(x, x_trans, model, p1=0.1, p2=0.1, iters=300):
    delta1 = (torch.randn(*x.shape) * 2 * p1 - p1).to(x.device)
    delta2 = (torch.randn(*x_trans.shape) * 2 * p2 - p2).to(x_trans.device)
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    
    for i in pbar:
        delta1.requires_grad_(True)
        delta2.requires_grad_(True)
        x_adv = x + delta1
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        x_emb = model.encode(x_adv).latent_dist.sample()
        x_dec = model.decode(x_emb).sample
        
        target = x_trans + delta2
        target.data = torch.clamp(target, min=-1.0, max=1.0)
        loss = criterion(x_dec, target)
        loss.backward()
        
        delta1 = delta1 - (2 / 255) * delta1.grad.sign()
        delta1 = torch.clamp(delta1, min=-p1, max=p1).detach_()
        delta2 = delta2 - (2 / 255) * delta2.grad.sign()
        delta2 = torch.clamp(delta2, min=-p2, max=p2).detach_()
        pbar.set_description(f"[Running poi]: Loss {loss.item():.5f}")

    with torch.no_grad():
        x_adv = x + delta1
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        target = x_trans + delta2
        target.data = torch.clamp(target, min=-1.0, max=1.0)
    return x_adv, target
