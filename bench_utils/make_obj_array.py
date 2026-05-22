import numpy as np
def make_obj_array(lst):
    out = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst):
        out[i] = x
    return out