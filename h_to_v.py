import numpy as np

def h_to_v(dbm, hidden):
    # Samples the visible nodes of the RBM conditional on the hidden nodes.

    # Calculate the activations of the visible units
    V = 1.0 / (1.0 + np.exp(-np.dot(hidden, dbm['w'][0].T) - dbm['b'][0].T))

    # Sample binary visible units based on probabilities
    Vrand = np.random.rand(*V.shape)
    #Vrand = 0.5*np.ones((V.shape[0], V.shape[1]))
    V = np.array(V >= Vrand, dtype=float)

    return V
