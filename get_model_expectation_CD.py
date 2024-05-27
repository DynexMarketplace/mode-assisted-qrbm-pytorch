import numpy as np
from h_to_v import *
from v_to_h import *
import copy
def get_model_expectation_CD(rbm, data, CDiters):
    # Calculate CD model expectation

    dataCount = data.shape[0]
    cd_vis = copy.deepcopy(data)

    for i in range(CDiters):
        cd_hid, _ = v_to_h(rbm, cd_vis)
        cd_vis = h_to_v(rbm, cd_hid)

    cd_hid, _ = v_to_h(rbm, cd_vis)

    model_expectation = np.dot(cd_vis.T, cd_hid) / dataCount
    model_vis_avg = (np.sum(cd_vis, axis=0) / dataCount).reshape(-1,1)
    model_hidden_avg = (np.sum(cd_hid, axis=0) / dataCount).reshape(-1,1)

    return model_expectation, model_vis_avg, model_hidden_avg, cd_vis