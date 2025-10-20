# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = torch.tensor(-20.0, device=device)
        self.lambda_max = torch.tensor(20.0, device=device)



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        u = t.to(torch.float32) /self.n_steps
        b = torch.atan(torch.exp(-self.lambda_max / 2))
        a = torch.atan(torch.exp(-self.lambda_min / 2)) - b
        phi = a * u + b
        lambda_t = -2.0 * torch.log(torch.tan(phi))
        return lambda_t.view(-1, 1, 1, 1)

    def alpha_lambda(self, lambda_t: torch.Tensor): 
        var = torch.sigmoid(lambda_t)

        return var.sqrt()
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        var = torch.sigmoid(-lambda_t)

        return var.sqrt()
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        alpha = self.alpha_lambda(lambda_t)
        sigma = self.sigma_lambda(lambda_t)
        z_lambda_t = alpha * x + sigma * noise

        return z_lambda_t
               
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        exp_ratio = torch.exp(lambda_t - lambda_t_prim)
        sigma2 = torch.sigmoid(-lambda_t)
        var_q = (1 - exp_ratio) * sigma2

        return var_q.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        exp_ratio = torch.exp(lambda_t - lambda_t_prim)
        sigma2_prim = torch.sigmoid(-lambda_t_prim)
        var_q_x = (1 - exp_ratio) * sigma2_prim

    
        return var_q_x.sqrt()

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        exp_ratio = torch.exp(lambda_t - lambda_t_prim)
        alpha_t = torch.sigmoid(lambda_t).sqrt()
        alpha_prim = torch.sigmoid(lambda_t_prim).sqrt()
        mu = (exp_ratio * (alpha_prim / alpha_t) * z_lambda_t
            + (1 - exp_ratio) * alpha_prim * x)
    
        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        exp_ratio = torch.exp(lambda_t - lambda_t_prim)
        sigma_q = ((1 - exp_ratio) * torch.sigmoid(-lambda_t)).sqrt()
        sigma_q_x = ((1 - exp_ratio) * torch.sigmoid(-lambda_t_prim)).sqrt()
        var = sigma_q_x.pow(2 * (1 - v)) * sigma_q.pow(2 * v)

        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)
        std = var.sqrt()
        eps = torch.randn_like(z_lambda_t)
        sample = mu + std * eps

    
        return sample 

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor = None, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        lambda_t = self.get_lambda(t)
        z = self.q_sample(x0, lambda_t, noise)
        eps_pred = self.eps_model(z, labels)
        diff = noise - eps_pred
        sq = diff.pow(2)
        loss = sq.sum(dim=dim).mean()

        return loss
