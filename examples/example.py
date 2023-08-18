#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl

import quimb.tensor as qtn

from tsp_misc_tns import make_aklt_mps, make_splitted_mps

from tsp import MPSPreparation 

if __name__ == "__main__":
    
    # TODO: Optimize two-qubit gates over isometric manifold.
    # TODO: QCTN (Quantum Circuit Tensor Networks (also works in 2D))
    # TODO: isoPEPS preparation not published yet.
    # TODO: MPS in Quimb Format -> Gates? (also for LCU circuits)    
    # TODO: Input and format of the MPS - should be framework agnostic
    # TODO: LCU to tket circuit
    # TODO: Break 4 qubit gate
    # TODO: 2d AKLT
    # TODO: sanity check for TSP - d==chi
    # add references
    # add documentation
      
    qubit_hamiltonian = 0
    mps_type = 'heisenberg'#'P4'#'aklt'#'aklt'#''random'#'random'#

    if mps_type == 'aklt':
        L=8
        tens, bond = make_aklt_mps(L)
        tens = make_splitted_mps(tens)
        target_mps = qtn.MatrixProductState(tens, shape='lrp')
        target_mps.normalize()
        

    if mps_type == 'random':
        L = 8
        target_mps = (qtn.MPS_rand_state(L=L, bond_dim=2) + 
                      qtn.MPS_rand_state(L=L, bond_dim=2)*0*1j)

        
    if mps_type in ['P4','N2','heisenberg']:
        filenames = {'P4': 'data/P4_6-31G_dist2.0000.pkl', 
                      'N2': 'data/N2_STO-6G_dist2.0000.pkl',
                      'heisenberg':'data/heisenberg_L32_dist0.8000.pkl'}
        
        with open(filenames[mps_type], 'rb') as f:
            data = pkl.load(f)
        
        target_mps = data['quimb_mps']
        target_mps.compress('right')
        target_mps.permute_arrays(shape='lpr')
        target_mps.normalize()
        qubit_hamiltonian = data['qubit_hamiltonian']
        

    
    mpsp = MPSPreparation(target_mps, qubit_hamiltonian=qubit_hamiltonian)
    # 
    number_of_layers = 4
    mpsp.seq_preparation(number_of_layers, do_compression=False, max_bond_dim=64, verbose=True)
    
    number_of_layers = 2
    mpsp.variational_seq_preparation(number_of_layers, do_compression=False, max_bond_dim=64, verbose=True)
    # 
    depth = 8
    mpsp.qctn_preparation(depth)
    
    number_of_lcu_layers = 4  
    mpsp.lcu_preparation(number_of_lcu_layers, verbose=False)
    
    number_of_lcu_layers = 4
    mpsp.variational_lcu_preparation(number_of_lcu_layers, verbose=False)
    
    
    # L=8
    # tens, bond = make_aklt_mps(L)
    # target_mps = qtn.MatrixProductState(tens, shape='lrp')
    # target_mps.normalize()
    
    # mpsp = MPSPreparation(target_mps)
    # mpsp.adiabatic_state_preparation()
    
    # # x, y = (mpsp.adiabatic_data['target_fidelity'].keys(), 
    # #         mpsp.adiabatic_data['target_fidelity'].values())
    # # plt.plot(x, y, '.-')
        
    
    # x, y = (mpsp.adiabatic_data['ss'].keys(), 
    #         mpsp.adiabatic_data['ss'].values())
    # plt.plot(x, y, '.-')      
    
    
