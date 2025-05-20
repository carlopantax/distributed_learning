import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, eta=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                param_norm = torch.norm(p)
                grad_norm = torch.norm(grad)

                if param_norm != 0 and grad_norm != 0:
                    local_lr = eta * param_norm / (grad_norm + weight_decay * param_norm + eps)
                else:
                    local_lr = 1.0

                # apply weight decay directly to the gradient
                if weight_decay != 0:
                    grad = grad + weight_decay * p

                if 'momentum_buffer' not in self.state[p]:
                    buf = self.state[p]['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                p.add_(buf, alpha=-group['lr'] * local_lr)