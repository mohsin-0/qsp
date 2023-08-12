#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
import numpy as np
import scipy as sp

from numpy import sin
half_pi = np.pi/2

from ncon import ncon 

from matplotlib import pyplot as plt

import quimb.tensor as qtn
import quimb as qu


def get_maps_for_square_lattice(Lx, Ly):
    
    
    pt2num = {(x, y): Lx*y + x for y in range(Ly) for x in range(Lx)}
    num2pt = {Lx*y + x: (x, y) for y in range(Ly) for x in range(Lx)}

    bonds = {}
    num2inds = {v: [None, None, None, None, f'p{v}'] for v in pt2num.values()}
    num2nns  = {v: [None, None, None, None] for v in pt2num.values()}   # site num to nearest neighbor

    for pt, v in pt2num.items():
        lft = pt2num[(np.mod(pt[0]-1, Lx), pt[1])]
        rht = pt2num[(np.mod(pt[0]+1, Lx), pt[1])]

        bot = pt2num[pt[0], (np.mod(pt[1]+1, Ly))]
        top = pt2num[pt[0], (np.mod(pt[1]-1, Ly))]

        num2inds[v  ][0] = f'b_{v}_{lft}'
        num2inds[lft][2] = f'b_{v}_{lft}'

        num2inds[v  ][1] = f'b_{v}_{bot}'
        num2inds[bot][3] = f'b_{v}_{bot}'
        
        num2nns[v  ][0] = lft
        num2nns[lft][2] = v
        
        num2nns[v  ][1] = bot
        num2nns[bot][3] = v
        
        bonds[(lft, v)] = 'H' # ordering of the bond is important
        bonds[(bot, v)] = 'V'

    sublattices = {}
    sublattices['a'] = [num for num, pt in num2pt.items() if np.mod(np.mod(pt[0], 2)+np.mod(pt[1], 2), 2) == 0]
    sublattices['b'] = [num for num, pt in num2pt.items() if np.mod(np.mod(pt[0], 2)+np.mod(pt[1], 2), 2) == 1]

    return num2pt, pt2num, num2inds, sublattices, bonds, num2nns


def print_vector(vec):
    n = int(np.log2(np.prod(vec.shape)))
    sub_indx_dims = [2]*n
    
    for indx in range(np.prod(sub_indx_dims)):
        inds = np.unravel_index(indx, sub_indx_dims)
        
        if np.abs(vec[indx])>1e-12:
            print(''.join([f'{i}' for i in inds]), vec[indx])
                    

def constuct_parent_hamiltonian(Q, Lx=2,Ly=2):
    
    peps = qtn.PEPS.rand(Lx=5, Ly=5, bond_dim=3, seed=666)
    
    kernel = sp.linalg.null_space(ncon([Q,Q],[(-1,-2,1,-5,-7),(1,-3,-4,-6,-8)]).reshape(2**6,16**2))
    ham_term = 0.    
    for it in range(kernel.shape[1]):
        v = kernel[:,it]
        ham_term += ncon((np.conj(v),v),([-1],[-2]))
    ham_term = ham_term.reshape((16,16,16,16))
    return ham_term
    
    # Q_0 = (Q[0:1, :, :])
    # Q_n = (Q[:, 0:1, :])
    
    # kernel = sp.linalg.null_space(ncon([Q_0,Q],[(-1,1,-3),(1,-2,-4)]).reshape(2,4**2))
    # ham_term_0 = 0.    
    # for it in range(kernel.shape[1]):
    #     v = kernel[:,it]
    #     ham_term_0 += ncon((np.conj(v),v),([-1],[-2]))    
    # ham_term_0 = ham_term_0.reshape((4,4,4,4))
    
    # kernel = sp.linalg.null_space(ncon([Q,Q_n],[(-1,1,-3),(1,-2,-4)]).reshape(2,4**2))
    # ham_term_n = 0.    
    # for it in range(kernel.shape[1]):
    #     v = kernel[:,it]
    #     ham_term_n += ncon((np.conj(v),v),([-1],[-2]))    
    # ham_term_n = ham_term_n.reshape((4,4,4,4))
    
    # if not cyclic:
    #     H = [qtn.Tensor(ham_term, inds=(f'k{i}',f'k{i+1}', f'b{i}',f'b{i+1}')) for i in range(L-1)]
    #     H[ 0] = qtn.Tensor(ham_term_0, inds=(f'k{0}',f'k{1}', f'b{0}',f'b{1}'))
    #     H[-1] = qtn.Tensor(ham_term_n, inds=(f'k{L-2}',f'k{L-1}', f'b{L-2}',f'b{L-1}')) 
        
    # return H, ham_term, ham_term_0, ham_term_n


if __name__ == "__main__":    
    print('prepare 2d aklt state\n')
    
    x = np.array(qu.pauli('x'))
    y = np.array(qu.pauli('y'))
    z = np.array(qu.pauli('z'))
    I = np.array(qu.pauli('I'))
    
    s_ttl = {}
    for p, label in zip([x,y,z],['x','y','z']):
        tmp = 0
        tmp += ncon([p,I,I,I],[(-1,-5),(-2,-6),(-3,-7),(-4,-8)])
        tmp += ncon([I,p,I,I],[(-1,-5),(-2,-6),(-3,-7),(-4,-8)])
        tmp += ncon([I,I,p,I],[(-1,-5),(-2,-6),(-3,-7),(-4,-8)])
        tmp += ncon([I,I,I,p],[(-1,-5),(-2,-6),(-3,-7),(-4,-8)])
        s_ttl[label] = tmp.reshape(16,16)
        
    s_sqr = (s_ttl['x']@s_ttl['x'] + 
             s_ttl['y']@s_ttl['y'] + 
             s_ttl['z']@s_ttl['z'] )
    
    s_sqr = s_sqr.reshape(16,16)

        
    proj_symm = 0
    vals, vecs = sp.linalg.eigh(s_sqr)
    vecs = np.real(vecs)    
    for vec_it, val in enumerate(vals):
        if np.allclose(val, 24.):
            vec = vecs[:,vec_it]
            
            proj_symm += ncon([vec, np.conj(vec)],[(-1,),(-2,)])
            s_val = (vec.T)@s_sqr@vec
            z_val = (vec.T)@s_ttl['z']@vec
            print(s_val, z_val)
            print_vector(vec)
            print('\n')
            
    aklt_tensor =  proj_symm.reshape([2]*4+[16])
    
    

    
    Lx, Ly = 4, 4
    tensor_grid = []
    for y in range(Ly):
        tensor_row = []
        for x in range(Lx):
            tensor = aklt_tensor.copy()
            bdry = []
            if x==0:
                tensor = ncon([tensor, np.array([1,0]).reshape(2,-1)],[(1,-2,-3,-4,-5),(1,-1)])            
                bdry.append('L')
                
            if x==(Lx-1):
                tensor = ncon([tensor, np.array([1,0]).reshape(2,-1)],[(-1,-2,3,-4,-5),(3,-3)])            
                bdry.append('R')
                
            if y==0:
                tensor = ncon([tensor, np.array([1,0]).reshape(2,-1)],[(-1,2,-3,-4,-5),(2,-2)])            
                bdry.append('B')
                
            if y==(Ly-1):
                tensor = ncon([tensor, np.array([1,0]).reshape(2,-1)],[(-1,-2,-3,4,-5),(4,-4)])            
                bdry.append('T')
                
            tensor_row.append(tensor)#.squeeze())
        
            print(f'({x},{y}), {bdry}')
        tensor_grid.append(tensor_row)
    

    peps = qtn.PEPS(tensor_grid, shape='ldrup')
    ham_term = constuct_parent_hamiltonian(aklt_tensor, Lx=2,Ly=2)
    
    
    H2 = {}
    for i in range(Lx-1):
        for j in range(Ly-1):
            H2[(i, j), (i + 1, j)] = qu.qu(ham_term)
            H2[(i, j), (i, j + 1)] = qu.qu(ham_term)
        
        
    parent_hamiltonian = qtn.LocalHam2D(Lx, Ly, H2=H2)
    # parent_hamiltonian.draw()
    exp_val = peps.compute_local_expectation(parent_hamiltonian.terms, max_bond=32)
    print(exp_val)
    
    su = qtn.SimpleUpdate(
        peps, 
        parent_hamiltonian,
        chi=32,  # boundary contraction bond dim for computing energy
        compute_energy_every=10,
        compute_energy_per_site=True,
        keep_best=True,
        )