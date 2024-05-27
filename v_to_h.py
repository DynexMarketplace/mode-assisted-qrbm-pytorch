import numpy as np
import copy

def v_to_h(dbm, data):
    # Samples the hidden state of an RBM given the visible state
    # according to the sigmoidal conditional probability dist.

    # Calculate the activations of the hidden units
    H = 1.0 / (1.0 + np.exp(-np.dot(data, dbm['w'][0]) - dbm['b'][1].T))

    # Copy the probabilities to H_probs
    H_probs = copy.deepcopy(H)

    # Sample binary hidden units based on probabilities
    Hrand = np.random.rand(*H.shape)
    #Hrand = 0.5*np.ones((H.shape[0], H.shape[1]))
    H = np.array(H >= Hrand, dtype=float)

    return H, H_probs
