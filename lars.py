import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.

    Paper: https://arxiv.org/abs/1708.03888
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0001,
                 eta=0.001, eps=1e-8, exclude_bias_and_norm=False):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            eps=eps,
            exclude_bias_and_norm=exclude_bias_and_norm
        )
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            eps = group['eps']
            exclude_bias_and_norm = group['exclude_bias_and_norm']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Skip bias and norm layers if configured
                if exclude_bias_and_norm and (len(p.shape) == 1 or p.ndim == 1):
                    # Use normal SGD update for bias and norm params
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(grad)

                        p.add_(buf, alpha=-lr)
                    else:
                        p.add_(grad, alpha=-lr)

                    continue

                # Apply LARS update for other parameters
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad.data)

                # Calculate local learning rate
                if param_norm != 0 and grad_norm != 0:
                    # Add weight decay to gradient
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)
                        grad_norm = torch.norm(grad.data)

                    # Calculate the local learning rate
                    local_lr = eta * param_norm / (grad_norm + eps)
                else:
                    local_lr = 1.0

                # Apply momentum and update
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)

                    p.add_(buf, alpha=-local_lr * lr)
                else:
                    p.add_(grad, alpha=-local_lr * lr)