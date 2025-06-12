import torch
from torch.optim.optimizer import Optimizer
import math


class LAMB(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                #nuova versione, prima non c'era nulla
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                """
                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom
                """
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = (exp_avg / bias_correction1) / denom
                
                """
                if wd != 0:
                    update += wd * p.data
                """
                if wd != 0:
                    update.add_(p.data, alpha=wd)

                """
                r1 = p.data.norm()
                r2 = update.norm()
                trust_ratio = r1 / r2 if r1 != 0 and r2 != 0 else 1.0
                """
                # Trust ratio calculation with additional stability
                w_norm = p.data.norm(2).clamp(min=eps)
                g_norm = update.norm(2).clamp(min=eps)
                trust_ratio = torch.where(
                    (w_norm > 0) & (g_norm > 0),
                    w_norm / g_norm,
                    torch.ones_like(w_norm)
                )
                
                p.data.add_(update, alpha=-lr * trust_ratio)