#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
import numpy as np
from scipy.optimize import minimize 

from tqdm import tqdm

from tsp import (apply_unitary_layers_on_wfn, 
                 apply_inverse_unitary_layer_on_wfn,
                 generate_unitaries)

from utils import (norm_mps_ovrlap, 
                   cl_zero_mps, 
                   unitaries_specs, 
                   unitaries_sanity_check,
                   compute_energy_expval)


def compress_copy(psi, max_bond):
    f = psi.copy(deep=True)
    f.right_canonize(normalize=True)
    f.compress(form='right', max_bond=max_bond)
    if max_bond==2:
        unitary, u_params = generate_bond_d_unitary(f, [])
    else:
        unitary, u_params = 0
    return f, unitary, u_params
            
            
def generate_bond_d_unitary(psi, u_params=[]):
    L = psi.L
    ####    
    D2_psi = psi.copy(deep=True)
    D2_psi.compress('right', max_bond=2)
    D2_psi.right_canonize(normalize=True)
    
    Gs_lst = generate_unitaries(D2_psi)
    if len(u_params)==0:
        u_params = np.round(np.random.random(size=((L-2)*2+0)),2)*2*np.pi
        
    D2_psi = apply_inverse_unitary_layer_on_wfn(Gs_lst, D2_psi)
    return Gs_lst, u_params            


def loss_for_kappa(al, psi, approx_psi, residual, qubit_hamiltonian=0):
    return -np.abs(norm_mps_ovrlap(psi, approx_psi + al[0]*residual))
    
    
def generate_lcu_for_mps(mps, qubit_hamiltonian, italic_D=4, do_compression=False, max_bond_dim=None, verbose=True, overlaps_gap=4):
    
    approx_mps_overlap = []
    overlaps, ent_entropies, energies = [], [], []      # <mps | encoded mps (i.e. wfn constructed from 00...0 by applying unitaries)>
    it_Ds, depths, gate_counts = [], [], []
    kappas, unitaries, unitaries_params = [], [], []
    schmidt_values = []
    encoded_mpss = []
    
    zero_wfn = cl_zero_mps(mps.L)
    
    mps.permute_arrays(shape='lpr')
    mps.right_canonize(normalize=True)    
    
    overlaps_trace = []
    for it in tqdm(range(italic_D)):        
        
        if it==0:
            approx_mps, unitary, u_params = compress_copy(mps, 2)
            curr_us, curr_us_params = [unitary], [u_params]
            kappa = 1
            
            encoded_mps = apply_unitary_layers_on_wfn(curr_us, zero_wfn)
            
        else:
            residual, unitary, u_params = compress_copy(mps-((mps.H@approx_mps)/(approx_mps.H@approx_mps))*approx_mps, 2)
            curr_us, curr_us_params = [unitary], [u_params]
                            
            ###
            result = minimize(loss_for_kappa, np.array([0.1]), args=(mps, approx_mps, residual, qubit_hamiltonian), method='BFGS')
            kappa = result.x[0]
            
            ###
            approx_mps = approx_mps + kappa*residual
            encoded_mps = encoded_mps + kappa*apply_unitary_layers_on_wfn(curr_us, zero_wfn)
                
            if do_compression:
                approx_mps.right_canonize()
                approx_mps.compress(max_bond=max_bond_dim)
            
            if verbose:
                print(f'{it=}, {max_bond_dim=}, approx_mps_D={np.max(approx_mps.bond_sizes())}, residual_psi_D={np.max(residual.bond_sizes())}\n')
            
            encoded_mps.compress()
            schmidt_values.append(encoded_mps.schmidt_values(int(np.argmax(encoded_mps.bond_sizes()))))
        
        kappas.append(kappa)
        
        for u in curr_us:
            unitaries_sanity_check(u)
            
        unitaries.append(curr_us)
        unitaries_params.append(curr_us_params)
        encoded_mpss.append(encoded_mps)
        
        ##
        approx_mps_overlap.append( norm_mps_ovrlap(mps, approx_mps ))
        overlap = norm_mps_ovrlap(mps, encoded_mps)
        overlaps_trace.append(overlap)
        
        if verbose:
            print(f'it={it+1}, {overlap=}, approx_mps_overlap - encoded_mps_overlap = {overlap - approx_mps_overlap[-1]:g}')
            
        if np.mod(it, overlaps_gap)==0 or (it+1)==italic_D:
            # ee = encoded_mps.entropy(encoded_mps.L//2)
            
            approx_energy = 0
            if qubit_hamiltonian!=0:
                approx_energy = compute_energy_expval(encoded_mps, qubit_hamiltonian)
                
            if verbose:
                print(f'it={it+1}, encoded_mps_energy={approx_energy:.10f}')
            
            it_Ds.append(it)
            overlaps.append(overlap)
            energies.append(np.real(approx_energy))
    
    # unitaries = []
    preparation_data = {'kappas': kappas,
                        'unitaries': unitaries,
                        'unitaries_params': unitaries_params, 
                        'approx_mps_overlap': approx_mps_overlap, 
                        'it_Ds': it_Ds,
                        'depths': depths, 
                        'gate_counts': gate_counts, 
                        'overlaps': overlaps, 
                        'ent_entropies': ent_entropies, 
                        'energies': energies,
                        'schmidt_values': schmidt_values
                        }
    
    return preparation_data


if __name__ == "__main__":
    pass