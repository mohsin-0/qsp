# -*- coding: utf-8 -*-
import numpy as np


def print_vector(vec):
    n = int(np.log2(np.prod(vec.shape)))
    sub_indx_dims = [2] * n

    for indx in range(np.prod(sub_indx_dims)):
        inds = np.unravel_index(indx, sub_indx_dims)

        if np.abs(vec[indx]) > 1e-12:
            print("".join([f"{i}" for i in inds]), vec[indx])
