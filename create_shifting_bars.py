import numpy as np

def ismember_row(a,b):
    # Get the unique row index
    _, rev = np.unique(np.concatenate((b,a)),axis=0,return_inverse=True)
    # Split the index
    a_rev = rev[len(b):]
    b_rev = rev[:len(b)]
    # Return the result:
    return np.isin(a_rev,b_rev)


def create_shifting_bars(totlength, barlength):
    # Generate phaseSpace as binary representations
    phaseSpace = np.array([list(bin(i)[2:].zfill(totlength)) for i in range(2**totlength-1,-1,-1)], dtype=int)

    dataSetConst = np.zeros((0, totlength), dtype=int)

    for i in range(1, totlength + 1):
        element = np.zeros(totlength, dtype=int)
        if i + barlength - 1 <= totlength:
            element[i - 1:i + barlength - 1] = 1
        else:
            element[i - 1:] = 1
            element[:((i + barlength) % totlength) - 1] = 1

        dataSetConst = np.vstack((dataSetConst, element))

    testing1 = ismember_row(phaseSpace,dataSetConst).astype(float).reshape(-1,1)
    probs = testing1 / np.sum(testing1)
    dataSet = np.tile(dataSetConst, (1, 1))

    return dataSet, probs
