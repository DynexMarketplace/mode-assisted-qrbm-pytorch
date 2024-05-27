import numpy as np
import numpy.matlib

def exactZ(RAND_RBM):
    # Calculates exact partition function, p(v), and argmax_v p(v) for (small) RBMs.

    vishid = RAND_RBM['w'][0].T
    hidbiases = RAND_RBM['b'][0].T
    visbiases = RAND_RBM['b'][1].T

    numdims, numhids = vishid.shape
    log_prob_vv = np.zeros((2**numhids, 1))

    mm = np.array([0,1]).reshape((-1,1))
    for jj in range(numhids - 1):
        mm1 = np.vstack((np.zeros((len(mm),1)),np.ones((len(mm),1))))
        mm = np.vstack((mm,mm))
        mm = np.hstack((mm1,mm))

    hh = mm.astype(np.float64)
    

    # Compute hh*vishid'
    vishid_transpose = vishid.T
    hh_vishid_transpose = np.dot(hh, vishid_transpose)

    # Compute ones(2^numhids, 1)*visbiases
    ones_array = np.ones((2**numhids, 1))
    ones_times_visbiases = np.dot(ones_array, visbiases)

    # Compute hh*vishid' + ones(2^numhids,1)*visbiases
    sum_terms = hh_vishid_transpose + ones_times_visbiases

    # Compute log(1+exp(hh*vishid' + ones(2^numhids,1)*visbiases))
    log_exp_terms = np.log(1 + np.exp(sum_terms))
    tt = np.sum(log_exp_terms, axis=1).reshape(-1,1)
    # Compute hidbiases' and add to the result
    hidbiases_transpose = hidbiases.T
    aa = np.dot(hh, hidbiases_transpose)
    log_prob_vv = aa + tt

    # Convert log_prob_vv to double precision (optional, as Python handles floating-point numbers by default)
    log_prob_vv = log_prob_vv.astype(np.float64)

    

    xdims = log_prob_vv.shape
    dim = np.argwhere(np.array(xdims) > 1)[0][0]

    alpha = np.max(log_prob_vv, axis=dim) - np.log(np.finfo(float).max) / 2
    repdims = np.ones_like(xdims)
    repdims[dim] = xdims[dim]
    aa = np.tile(alpha, xdims)
    logZZ_true = alpha + np.log(np.sum(np.exp(log_prob_vv - aa), axis=dim))

    log_prob_vv = np.flip(log_prob_vv) - logZZ_true

    indx = np.argmax(np.flip(log_prob_vv))
    mode_v = hh[indx].reshape(1,-1)

    return log_prob_vv, logZZ_true, mode_v

# Example usage:
# log_prob_vv, logZZ_true, mode_v = exactZ(RAND_RBM)