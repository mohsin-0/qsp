#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize 

from tqdm import tqdm

import tsp
import tsp_helper_routines as tsp_hr

def compress_copy(psi, max_bond):
    f = psi.copy(deep=True)
    f.right_canonize(normalize=True)
    f.compress(form='right', max_bond=max_bond)
    
    if max_bond==2:
        unitary = tsp.generate_bond_d_unitary(f)
    else:
        unitary = 0
    return f, unitary
            
            
def loss_for_kappa(al, psi, approx_psi, residual, qubit_hamiltonian=0):
    return -np.abs(tsp_hr.norm_mps_ovrlap(psi, approx_psi + al[0]*residual))
    
    
def generate_lcu_for_mps(mps, 
                         italic_D=4, 
                         do_compression=False, 
                         max_bond_dim=None, 
                         qubit_hamiltonian=0, 
                         verbose=True, 
                         overlaps_gap=4):
    
    # <mps | encoded mps (i.e. wfn constructed from 00...0 by applying unitaries)>
    overlaps = []          
    approx_mps_overlap = []
    energies = []
    
    it_Ds, depths, gate_counts = [], [], []
    kappas, unitaries, = [], []
    
    zero_wfn = tsp_hr.cl_zero_mps(mps.L)
    
    mps.permute_arrays(shape='lpr')
    mps.right_canonize(normalize=True)    
    
    for it in tqdm(range(italic_D)):
        if it==0:
            approx_mps, unitary = compress_copy(mps, 2)
            curr_us = [unitary]
            kappa = 1
            
            encoded_mps = tsp.apply_unitary_layers_on_wfn(curr_us, zero_wfn)
            
        else:
            renom_factor = ((mps.H@approx_mps)/(approx_mps.H@approx_mps))
            residual, unitary = compress_copy(mps-renom_factor*approx_mps, 2)
            curr_us = [unitary]
                            
            ###
            result = minimize(loss_for_kappa, 
                              np.array([0.1]), 
                              args=(mps, approx_mps, residual, qubit_hamiltonian), 
                              method='BFGS')
            
            kappa = result.x[0]
            
            ###
            approx_mps = approx_mps + kappa*residual
            encoded_mps = encoded_mps + kappa*tsp.apply_unitary_layers_on_wfn(curr_us, zero_wfn)
                
            if do_compression:
                approx_mps.right_canonize()
                approx_mps.compress(max_bond=max_bond_dim)
            
            if verbose:
                print(f'{it=}, {max_bond_dim=}, '
                      'approx_mps_D = {np.max(approx_mps.bond_sizes())}, '
                      'residual_psi_D = {np.max(residual.bond_sizes())}\n')
            
            encoded_mps.compress()
        
        kappas.append(kappa)
        
        for u in curr_us:
            tsp_hr.unitaries_sanity_check(u)
            
        unitaries.append(curr_us)
        
        ##
        approx_mps_overlap.append( tsp_hr.norm_mps_ovrlap(mps, approx_mps ))
        overlap = tsp_hr.norm_mps_ovrlap(mps, encoded_mps)
        
        if verbose:
            print(f'it={it+1}, {overlap=}, '
                  'approx_mps_overlap - encoded_mps_overlap={overlap - approx_mps_overlap[-1]:g}')
            
        if np.mod(it, overlaps_gap)==0 or (it+1)==italic_D:
            # ee = encoded_mps.entropy(encoded_mps.L//2)
            
            approx_energy = 0
            if qubit_hamiltonian!=0:
                approx_energy = tsp_hr.compute_energy_expval(encoded_mps, qubit_hamiltonian)
                
            if verbose:
                print(f'it={it+1}, encoded_mps_energy={approx_energy:.10f}')
            
            it_Ds.append(it)
            overlaps.append(overlap)
            energies.append(np.real(approx_energy))
            
    # unitaries = []
    preparation_data = {'kappas': kappas,
                        'unitaries': unitaries,
                        'approx_mps_overlap': approx_mps_overlap, 
                        'it_Ds': it_Ds,
                        'depths': depths, 
                        'gate_counts': gate_counts, 
                        'overlaps': overlaps, 
                        'energies': energies }
    
    return preparation_data


if __name__ == "__main__":
    pass