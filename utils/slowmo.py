import copy
import torch

def initialize_slowmo_state(model):
    slowmo_buffer = {}
    for name, param in model.state_dict().items():
        if torch.is_floating_point(param):
            slowmo_buffer[name] = param.clone().zero_()
    return slowmo_buffer

def slowmo_update(global_model, averaged_weights, slowmo_buffer, beta=0.9, slowmo_lr=1.0):
    updated_state_dict = copy.deepcopy(global_model.state_dict())
    for name in updated_state_dict.keys():
        if name in slowmo_buffer:
            delta = averaged_weights[name] - global_model.state_dict()[name]
            slowmo_buffer[name] = beta * slowmo_buffer[name] + delta
            updated_state_dict[name] = global_model.state_dict()[name] + slowmo_lr * slowmo_buffer[name]
        else:
            updated_state_dict[name] = averaged_weights[name]
    return updated_state_dict, slowmo_buffer
