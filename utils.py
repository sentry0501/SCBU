import numpy as np

def rebin(arr, factor):
    f = (np.asarray(factor) - arr.shape) % factor
    temp = np.pad(arr, ((0, f[0]), (0, f[1])), 'edge')
    sh = temp.shape[0] // factor[0], factor[0], -1, factor[1]
    res = temp.reshape(sh).mean(-1).mean(1)
    return res[:res.shape[0] - f[0], : res.shape[1] - f[1]]

def rebinMax(arr, factor):
    f = (np.asarray(factor) - arr.shape) % factor
    temp = np.pad(arr, ((0, f[0]), (0, f[1])), 'edge')
    sh = temp.shape[0] // factor[0], factor[0], -1, factor[1]
    res = temp.reshape(sh).max(-1).max(1)
    return res[:res.shape[0] - f[0], : res.shape[1] - f[1]]