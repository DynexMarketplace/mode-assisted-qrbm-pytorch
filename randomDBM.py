import numpy as np

def randomDBM(nodes):
    # Defines a randomly initialized RBM as a dictionary of arrays
    # nodes = [visible layer size, hidden layer size]

    # Variance of Normal distribution
    scale = 0.01

    dbm = {}
    dbm['w'] = []
    dbm['b'] = []
    for i in range(len(nodes) - 1):
        dbm['w'].append(scale * np.random.randn(nodes[i], nodes[i+1]))
        dbm['b'].append(scale * np.random.randn(nodes[i], 1))
        #dbm['w'].append(scale * np.ones((nodes[i], nodes[i+1])))
        #dbm['b'].append(scale * np.ones((nodes[i], 1)))
        if i == len(nodes) - 2:
            dbm['b'].append(scale * np.random.randn(nodes[i+1], 1))
            #dbm['b'].append(scale * np.ones((nodes[i+1], 1)))
    return dbm

# Example usage:
# nodes = [visible_layer_size, hidden_layer_size]


## np.random.randn
