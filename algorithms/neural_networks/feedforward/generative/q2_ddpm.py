import torch 
from torch import nn 
from typing import Optional, Tuple


class DenoiseDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta


    ### UTILS
    def gather(self, c: torch.Tensor, t: torch.Tensor):
        c_ = c.gather(-1, t)
        return c_.reshape(-1, 1, 1, 1)

    ### FORWARD SAMPLING
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar_t = self.gather(self.alpha_bar, t)
        mean = torch.sqrt(alpha_bar_t) * x0
        var = self.gather(1.0 - self.alpha_bar, t)

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        std = torch.sqrt(var)
        sample = mean + std * eps

        return sample

    ### REVERSE SAMPLING
    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor):
        beta_t = self.gather(self.beta, t)
        alpha_t = self.gather(self.alpha, t)
        alpha_bar_t = self.gather(self.alpha_bar, t)
        eps_theta = self.eps_model(xt, t)
        mu_theta = (1.0 / torch.sqrt(alpha_t)) * (
            xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_theta
        )
        var = beta_t

        return mu_theta, var

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        mu, var = self.p_xt_prev_xt(xt, t)
        std = torch.sqrt(var)
        # noise only for t>0
        eps = torch.randn_like(xt)
        mask = (t.view(-1, 1, 1, 1) > 0).float()
        sample = mu + std * eps * mask

        return sample

    ### LOSS
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        # dim = list(range(1, x0.ndim))
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_theta = self.eps_model(x_t, t)
        sq_error = (noise - eps_theta) ** 2
        dims = list(range(1, sq_error.ndim))

        loss = sq_error.sum(dim=dims).mean()

        return loss
