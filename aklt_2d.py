#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt

import pickle as pkl
import numpy as np
import scipy as sp

from numpy import sin
half_pi = np.pi/2

from ncon import ncon 

import quimb.tensor as qtn
import quimb as qu


def get_maps_for_square_lattice(Lx, Ly):
    
    pt2num = {(x, y): Lx*y + x for y in range(Ly) for x in range(Lx)}
    num2pt = {Lx*y + x: (x, y) for y in range(Ly) for x in range(Lx)}
    return num2pt, pt2num


def print_vector(vec):
    n = int(np.log2(np.prod(vec.shape)))
    sub_indx_dims = [2]*n
    
    for indx in range(np.prod(sub_indx_dims)):
        inds = np.unravel_index(indx, sub_indx_dims)
        
        if np.abs(vec[indx])>1e-12:
            print(''.join([f'{i}' for i in inds]), vec[indx])
                    
        
def construct_aklt_tensor():
    # print('prepare 2d aklt state\n')
    
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
            # print(s_val, z_val)
            # print_vector(vec)
            # print('\n')
            
    proj_symm =  proj_symm.reshape([2]*4+[16])
    singlet = np.sqrt(0.5) * np.array([[0., -1.], [1.,  0.]]) 
    aklt_tensor = ncon([proj_symm, singlet, singlet], [(-1,-2,3,4,-5),(-3,3),(-4,4)])

    return aklt_tensor


def construct_parent_hamiltonian(tensor_grid, bonds, num2pt, Lx, Ly):
    # print('constuct parent hamiltonian with open bc')
    hamiltonian  = {}
    
    H2 = {}
    for bond in bonds:
        (x1,y1), (x2,y2), orientation = num2pt[bond[0]], num2pt[bond[1]], bond[2]
        Q1 = tensor_grid[y1][x1]
        Q2 = tensor_grid[y2][x2]
        
        # print((x1,y1), (x2,y2), orientation, Q1.shape, Q2.shape)
        
        if orientation=='H':
            kernel = sp.linalg.null_space(ncon([Q1,Q2],[(-1,-2,1,-5,-7),(1,-3,-4,-6,-8)]).reshape(-1,16**2))
            
        if orientation=='V':
            kernel = sp.linalg.null_space(ncon([Q1,Q2],[(-1,-3,-4,1,-7),(-2,1,-5,-6,-8)]).reshape(-1,16**2))
            
        ham_term = 0.
        for it in range(kernel.shape[1]):
            v = kernel[:,it]
            ham_term += ncon((np.conj(v),v),([-1],[-2]))            
        
        ham_term = ham_term.reshape((16,16,16,16))
        hamiltonian[bond] = ham_term
        
        H2[(y1,x1), (y2, x2)] = qu.qu(ham_term)

                
    quimb_hamiltonian = qtn.LocalHam2D(Lx, Ly, H2=H2)
    # quimb_hamiltonian.draw()

    return hamiltonian, quimb_hamiltonian


def construct_tensor_grid(local_tensor, pt2num, Lx, Ly):
    tensor_grid = []
    bonds = set()
    for y in range(Ly):
        tensor_row = []
        for x in range(Lx):
            
            if x<(Lx-1):
                bonds.add(( pt2num[(x,y)], pt2num[(x+1,y)], 'H'))
                
            if y<(Ly-1):
                bonds.add(( pt2num[(x,y)], pt2num[(x,y+1)], 'V'))
                
            tensor = local_tensor.copy()
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
        
            # print(f'({x},{y}), {bdry}')
        tensor_grid.append(tensor_row)
        
    return tensor_grid, bonds


def apply_evolution_opeerator(parent_hamiltonian, peps, tau, Tmax):
    d = 16
    max_bond = 4
    for bond, curr_term in parent_hamiltonian.terms.items():
        # print(bond, curr_term.shape)
        curr_gate = sp.linalg.expm(curr_term.reshape(d**2,d**2)*-1j*(tau/1)*Tmax).reshape((d,d,d,d))
        peps.gate(curr_gate, where=bond, contract='split', inplace=True)
        
        peps.compress_all(inplace=True, max_bond=max_bond)
        


if __name__ == "__main__":    

    Lx, Ly = 2, 6
    num2pt, pt2num = get_maps_for_square_lattice(Lx, Ly)
    
    aklt_tensor = construct_aklt_tensor()
    bell_tensor = (np.eye(16)).reshape(aklt_tensor.shape)
    
    tensor_grid, bonds = construct_tensor_grid(aklt_tensor, pt2num, Lx, Ly)
    aklt_peps = qtn.PEPS(tensor_grid, shape='ldrup')
    aklt_peps = aklt_peps/np.sqrt(aklt_peps.H@aklt_peps)
    
    tensor_grid, _ = construct_tensor_grid(bell_tensor, pt2num, Lx, Ly)
    bell_peps = qtn.PEPS(tensor_grid, shape='ldrup')
    
    
    T, tau = 6, 0.04
    # s_func = lambda t,T=T: sin( half_pi*sin(half_pi*t/T)**2 )**2
    s_func = lambda t,T=T: sin(half_pi*t/T)**2
    # s_func = lambda t,T=T: t/T
    
    ts = np.arange(0, T+tau, tau)
    
    peps = bell_peps
    for t_it, t in enumerate(ts):
        s = s_func(t)
        local_tensor = (1-s)*bell_tensor + s*aklt_tensor
        
        tensor_grid, bonds = construct_tensor_grid(local_tensor, pt2num, Lx, Ly)    
        _, parent_hamiltonian = construct_parent_hamiltonian(tensor_grid, bonds, num2pt, Lx=Lx, Ly=Ly)
        
        apply_evolution_opeerator(parent_hamiltonian, peps, tau=tau, Tmax=T)
        peps = peps/np.sqrt(peps.H@peps)
        
        
        # peps = qtn.PEPS(tensor_grid, shape='ldrup')
        exp_val = np.real(peps.compute_local_expectation(parent_hamiltonian.terms, max_bond=32))
        # denom = np.sqrt((peps.H@peps)*(aklt_peps.H@aklt_peps))
        target_fidelity = np.abs(peps.H@
                                 aklt_peps)#/denom)
        
        print(f'{t_it=:03d}, {t=:.2f}, {s=:.5f}, {exp_val=:.08f}, {target_fidelity=:.08f}')
        
