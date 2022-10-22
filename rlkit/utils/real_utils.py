import numpy as np

def filter_step_fn(ob, action, next_ob, min_action_value=0.3):
    for dim in [0, 1, 2, 4]:
        if np.abs(action[dim]) > min_action_value:
            return False
    return True
