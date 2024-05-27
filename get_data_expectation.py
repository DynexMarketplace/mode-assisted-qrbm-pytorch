import numpy as np
from v_to_h import *
def get_data_expectation(rbm, data):
    # Computes data expectation of the RBM log-likelihood gradient

    dataCount = data.shape[0]

    # Calculate hidden activations and probabilities
    hid0, h_probs = v_to_h(rbm, data)
    vis0 = data

    # Use 1s and 0s instead of probabilities for h_probs
    h_probs = hid0

    # Data expectation calculation
    data_expectation = np.dot(vis0.T, h_probs) / dataCount
    data_hidden_avg = (np.sum(h_probs, axis=0) / dataCount).reshape(-1,1)
    data_vis_avg = (np.sum(vis0, axis=0) / dataCount).reshape(-1,1)

    return data_expectation, data_vis_avg, data_hidden_avg
