#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

import tsp
import tsp_helper_routines as tsp_hr


def generate_sequ_for_mps(mps, 
                          italic_D, 
                          do_compression=False, 
                          max_bond_dim=None, 
                          qubit_hamiltonian=0,
                          verbose=True, 
                          overlaps_gap=4):
    
    # <disentangled_mps | 00...0>
    disentangled_mps_overlaps = []
    
    # <mps | encoded mps (i.e. wfn constructed from 00...0 by applying unitaries)>
    overlaps = []                       
    
    it_Ds, depths, gate_counts = [], [], []
    energies, unitaries = [], []
    
    mps.permute_arrays(shape='lpr')
    mps.right_canonize(normalize=True)
    mps_orig = mps.copy(deep=True)
    
    zero_wfn = tsp_hr.cl_zero_mps(mps.L)
    depth, gate_count = 0, 0
    
    for it in tqdm(range(italic_D)):        
        unitary = tsp.generate_bond_d_unitary(mps)

        tsp_hr.unitaries_sanity_check(unitary)
        unitaries.append(unitary)
        
        mps = tsp.apply_inverse_unitary_layer_on_wfn(unitary, mps)
        if do_compression:
            mps.right_canonize(normalize=True)
            mps.compress('right', max_bond=max_bond_dim)
        
        mps.right_canonize(normalize=True)
        mps.compress()
        
        ##
        disentangled_mps_overlaps.append(tsp_hr.norm_mps_ovrlap(mps, zero_wfn))
        
        ##
        u_depth, u_gate_count = tsp_hr.unitaries_specs(unitary)
        depth += u_depth
        gate_count += u_gate_count
        
        if np.mod(it, overlaps_gap)==0 or (it+1)==italic_D:
            encoded_mps = tsp.apply_unitary_layers_on_wfn(unitaries, zero_wfn)
            encoded_mps.right_canonize(normalize=True)
            overlap = tsp_hr.norm_mps_ovrlap(encoded_mps, mps_orig)
            
            if verbose:
                print(f'it={it+1}, encoded_mps_overlap={np.abs(overlap):.10f}')
            
            approx_energy  = 0
            if qubit_hamiltonian!=0:
                approx_energy = tsp_hr.compute_energy_expval(encoded_mps, qubit_hamiltonian)
                if verbose:    
                    print(f'it={it+1}, encoded_mps_energy={approx_energy:.10f}')
            
            it_Ds.append(it)
            overlaps.append(overlap)
            energies.append(approx_energy)
            depths.append(depth)
            gate_counts.append(gate_count)
            
            
    preparation_data = {'disentangled_mps_overlaps': disentangled_mps_overlaps, 
                        'overlaps': overlaps, 
                        'it_Ds': it_Ds, 
                        'depths': depths, 
                        'gate_counts': gate_counts, 
                        'energies': energies,
                        'unitaries': unitaries}
    
    return preparation_data


if __name__ == "__main__":
    pass