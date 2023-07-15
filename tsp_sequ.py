#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from tsp import (apply_unitary_layers_on_wfn, 
                 apply_inverse_unitary_layer_on_wfn,
                 generate_unitaries)

from utils import (norm_mps_ovrlap, 
                   cl_zero_mps, 
                   unitaries_specs, 
                   unitaries_sanity_check,
                   compute_energy_expval)


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


def generate_sequ_for_mps(mps, qubit_hamiltonian, italic_D, do_compression=False, max_bond_dim=None, verbose=True, overlaps_gap = 4):
    disentg_mps_overlaps = []           # <disentangled_mps | 00...0>
    overlaps = []                       # <mps | encoded mps (i.e. wfn constructed from 00...0 by applying unitaries)>
    it_Ds, depths, gate_counts = [], [], []
    ent_entropies, energies, unitaries, unitaries_params = [], [], [], []
    schmidt_values = []
    
    mps.permute_arrays(shape='lpr')
    mps.right_canonize(normalize=True)
    mps_orig = mps.copy(deep=True)
    
    zero_wfn = cl_zero_mps(mps.L)
    depth, gate_count = 0, 0
    
    for it in tqdm(range(italic_D)):        
        unitary, u_params = generate_bond_d_unitary(mps, [])

        unitaries_sanity_check(unitary)
        unitaries.append(unitary)
        unitaries_params.append(u_params)

        mps = apply_inverse_unitary_layer_on_wfn(unitary, mps)#MatrixProductState.from_dense( apply_inverse_unitary_layer_on_wfn(unitary, mps).to_dense(), dims=[2]*mps.L)
        if do_compression:
            mps.right_canonize(normalize=True)
            mps.compress('right', max_bond=max_bond_dim)
        
        mps.right_canonize(normalize=True)
        
        mps.compress()
        schmidt_values.append(mps.schmidt_values(int(np.argmax(mps.bond_sizes()))))
        
        ##
        disentg_mps_overlaps.append(norm_mps_ovrlap(mps, zero_wfn))
        
        ##
        u_depth, u_gate_count=unitaries_specs(unitary)
        depth += u_depth
        gate_count += u_gate_count
        
        if np.mod(it, overlaps_gap)==0 or (it+1)==italic_D:
            encoded_mps = apply_unitary_layers_on_wfn(unitaries, zero_wfn)
            encoded_mps.right_canonize(normalize=True)
            overlap = norm_mps_ovrlap(encoded_mps, mps_orig)
            ee = encoded_mps.entropy(encoded_mps.L//2)
            if verbose:
                print(f'it = {it+1}, encoded_mps_overlap  = {np.abs(overlap):.10f}, ee = {ee:.10f}')
            
            approx_energy  = 0
            if qubit_hamiltonian!=0:
                approx_energy = compute_energy_expval(encoded_mps, qubit_hamiltonian)
                if verbose:    
                    print(f'it = {it+1}, encoded_mps_energy= {approx_energy:.10f}')
            
            it_Ds.append(it)
            overlaps.append(overlap)
            ent_entropies.append(ee)
            energies.append(approx_energy)
            depths.append(depth)
            gate_counts.append(gate_count)
            
            
    preparation_data = {'disentg_mps_overlaps': disentg_mps_overlaps, 
                        'overlaps': overlaps, 
                        'it_Ds': it_Ds, 
                        'depths': depths, 
                        'gate_counts': gate_counts, 
                        'ent_entropies': ent_entropies, 
                        'energies': energies,
                        'unitaries': unitaries, 
                        'unitaries_params': unitaries_params,
                        'schmidt_values': schmidt_values
                        }
    
    return preparation_data


if __name__ == "__main__":
    pass