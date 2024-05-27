import numpy as np
import dimod
from dbn2qubo import *

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
        
def rbm2qubo(rbm):
    # Extract the parameters from the RBM dictionary
    vishid = rbm['w'][0]
    hidbiases = rbm['b'][0].flatten()
    visbiases = rbm['b'][1].flatten()
    
    # Number of visible and hidden nodes
    n_visible = vishid.shape[0]
    n_hidden = vishid.shape[1]
    
    # Initialize the QUBO matrix with zeros
    n_total = n_visible + n_hidden
    Q = np.zeros((n_total, n_total))
    
    # Populate the diagonal entries with biases
    Q[:n_visible, :n_visible] = np.diag(hidbiases)
    Q[n_visible:, n_visible:] = np.diag(visbiases)
    
    # Populate the off-diagonal entries with weights
    Q[:n_visible, n_visible:] = vishid
    # Q[n_visible:, :n_visible] = -vishid.T
    
    return Q
    
def get_model_expectation_mode(rbm, model_type='rbm', solver_type='Exact'):
    """Compute model expectations using either RBM or DBN and different solvers.
    
    Parameters:
    - rbm (dict): The restricted Boltzmann machine model.
    - model_type (str): Type of the model to use ('rbm' or 'dbn').
    - solver_type (str): Type of the solver to use ('Exact' or 'Sampler').
    
    Returns:
    - tuple: Model expectations, visible layer averages, hidden layer averages, and mode push value.
    """

    # Extract node dimensions from the RBM
    visible_nodes, hidden_nodes = rbm['w'][0].shape
    
    # Calculate the full weight size
    fullWsize = np.prod(np.array([visible_nodes, hidden_nodes]) + 1)

    # Convert the RBM/DBN to a QUBO model based on user input
    if model_type == 'rbm':
        Q = rbm2qubo(rbm)
    elif model_type == 'dbn':
        Q = dbn2qubo(rbm, [visible_nodes, hidden_nodes])
    else:
        raise ValueError("Invalid model_type. Choose either 'rbm' or 'dbn'.")

    # Choose the solver based on user input
    if solver_type == 'Exact':
        sampler = dimod.ExactSolver()
    elif solver_type == 'Sampler':
        sampler = dimod.SimulatedAnnealingSampler()
    else:
        raise ValueError("Invalid solver_type. Choose either 'Exact' or 'Sampler'.")
        
    # Parameters to optimize training speed, tailored to RBM model sizes
    simulated_annealing_parameters = {
        'beta_range': [0.1, 1.0],
        'num_reads': 4,
        'num_sweeps': 25
    }
    # Sample the QUBO to find the ground state
    response = sampler.sample_qubo(-Q, **simulated_annealing_parameters)
    ground_state = response.first.sample
    ground_state_energy = response.first.energy

    # Extract visible and hidden layer states from the ground state
    mode_v = np.array([ground_state[i] for i in range(visible_nodes)]).reshape(1, -1)
    mode_h = np.array([ground_state[i] for i in range(visible_nodes, visible_nodes + hidden_nodes)]).reshape(1, -1)

    # Calculate model expectations and averages
    model_expectation = np.dot(mode_v.T, mode_h)
    model_vis_avg = mode_v.T
    model_hidden_avg = mode_h.T

    # Calculate the mode push, derived from supplementary material
    mode_push = (1 / (4 * fullWsize)) * (-ground_state_energy - 0.5 * np.sum(rbm['b'][0]) - 0.5 * np.sum(rbm['b'][1]) - (1 / 4) * np.sum(rbm['w'][0]))
    #mode_push = (4*fullWsize)^(-1)*(-ground_state_energy - 0.5*sum(rbm.b{1,1}) - 0.5*sum(rbm.b{2,1}) - (1/4)*sum(rbm.W{1,1}(:)));
    
    return model_expectation, model_vis_avg, model_hidden_avg, mode_push
    
# Example usage:
# model_expectation, model_vis_avg, model_hidden_avg, mode_push = get_model_expectation_mode(rbm, model_type='rbm', solver_type='Exact')
