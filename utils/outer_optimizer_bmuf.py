import copy

def initialize_global_momentum(model):
    """
    Initialize momentum buffer for BMUF.
    """
    momentum_buffer = {}
    for name, param in model.state_dict().items():
        momentum_buffer[name] = param.clone().zero_()
    return momentum_buffer

def global_momentum_update(global_model, averaged_weights, momentum_buffer, beta=0.9):
    """
    Apply BMUF-style global momentum update.
    """
    updated_state_dict = copy.deepcopy(global_model.state_dict())

    for name in updated_state_dict.keys():
        if name in momentum_buffer:
            delta = averaged_weights[name] - global_model.state_dict()[name]
            momentum_buffer[name] = beta * momentum_buffer[name] + delta
            updated_state_dict[name] = global_model.state_dict()[name] + momentum_buffer[name]
        else:
            updated_state_dict[name] = averaged_weights[name]

    return updated_state_dict, momentum_buffer
