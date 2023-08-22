#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pickle as pkl

with open('evo.pkl', 'rb') as f:
    evo = pkl.load(f)
n = 4
p = 2**(n-1)

[u1,cs,u2] = sp.linalg.cossin(evo, p=p, q=p)
A1, B1 = u1[:p,:p], u1[p:,p:]
A2, B2 = u2[:p,:p], u2[p:,p:]

C, S = cs[:p,:p], cs[:p,p:]


########################################################
u1 = np.zeros_like(u1)
cs = np.zeros_like(cs)
u2 = np.zeros_like(u2)

u1[:p,:p] = A1
u1[p:,p:] = B1

u2[:p,:p] = A2
u2[p:,p:] = B2

cs[:p,:p] = C
cs[p:,p:] = C

cs[:p,p:] = S
cs[p:,:p] =-S

assert np.allclose(u1@cs@u2-evo, np.zeros_like(u1))
########################################################