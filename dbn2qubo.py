import numpy as np

def dbn2qubo(dbn, nodes):
    # Returns Q matrix defined by RBM energy E = -x'Qx
    Qmatrix = np.zeros((np.sum(nodes), np.sum(nodes)))

    LayerNum = len(dbn['w']) + 1

    layerIndx = 0
    for i in range(1, LayerNum + 1):
        if i < LayerNum:
            Qmatrix[layerIndx:(layerIndx + nodes[i - 1]), layerIndx:(layerIndx + nodes[i - 1])] = np.diag(dbn['b'][i - 1].flatten())
            Qmatrix[layerIndx:(layerIndx + nodes[i - 1]), layerIndx + nodes[i - 1]:layerIndx + nodes[i - 1] + nodes[i]] = dbn['w'][i - 1]
            layerIndx += nodes[i - 1]
        elif i == LayerNum:
            Qmatrix[layerIndx:(layerIndx + nodes[i - 1]), layerIndx:(layerIndx + nodes[i - 1])] = np.diag(dbn['b'][i - 1].flatten())

    return Qmatrix
