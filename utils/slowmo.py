import copy
import torch

def initialize_slowmo_state(model):
    """
    Initialize the SlowMo server-side momentum buffer.

    For every floating-point tensor in the global model's state_dict, create a
    zero-initialized buffer that will accumulate an exponentially decayed
    direction of update across synchronizations.

    Args:
        model: Current global model.

    Returns:
         Mapping param_name -> momentum buffer
    """
    slowmo_buffer = {}
    for name, param in model.state_dict().items():
        if torch.is_floating_point(param):
            slowmo_buffer[name] = param.clone().zero_()
    return slowmo_buffer

def slowmo_update(global_model, averaged_weights, slowmo_buffer, beta=0.9, slowmo_lr=1.0):
    """
    Apply the SlowMo update to the global model.

    Computes the increment between the averaged client weights and the current
    global weights, updates the server-side momentum buffer, and applies a
    "slow" step to the global model:

        delta       = averaged_weights - global_weights
        m_t         = beta * m_{t-1} + delta
        global_new  = global_old + slowmo_lr * m_t

    where:
      - beta (0 < beta < 1) is the exponential momentum coefficient,
      - slowmo_lr controls the magnitude of the SlowMo correction step
        (typically in [0, 1]).

    Args:
        global_model: Global model before the update.
        averaged_weights: Client-averaged weights.
        slowmo_buffer: Server-side momentum buffer.
        beta: Momentum decay factor.
        slowmo_lr: SlowMo correction step applied to the global model.

    Returns:
        - Updated state_dict for the global model
        - Updated slowmo_buffer
    """
    updated_state_dict = copy.deepcopy(global_model.state_dict())
    for name in updated_state_dict.keys():
        if name in slowmo_buffer:
            delta = averaged_weights[name] - global_model.state_dict()[name]
            slowmo_buffer[name] = beta * slowmo_buffer[name] + delta
            updated_state_dict[name] = global_model.state_dict()[name] + slowmo_lr * slowmo_buffer[name]
        else:
            updated_state_dict[name] = averaged_weights[name]
    return updated_state_dict, slowmo_buffer
