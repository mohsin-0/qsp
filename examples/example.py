#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl

import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(__file__)).strip('examples')+'scr')

import quimb.tensor as qtn

from tsp_misc_tns import make_aklt_mps, make_splitted_mps
from tsp_misc_tns import make_aklt_peps

from tsp import MPSPreparation, PEPSPreparation

if __name__ == "__main__":
    
    # TODO: Optimize two-qubit gates over isometric manifold.
    # TODO: QCTN for  2D))
    # TODO: isoPEPS preparation not published yet.
    # TODO: MPS in Quimb Format -> Gates? (also for LCU circuits)    
    # TODO: Input and format of the MPS - should be framework agnostic
    # TODO: sanity check for TSP - d==chi
    # TODO: ***LCU to tket circuit
    # add references
    # add documentation
      
    qubit_hamiltonian = 0
    mps_type = 'random'#'heisenberg'#'P4'#'aklt'#

    if mps_type == 'aklt':
        L = 8
        tens, bond = make_aklt_mps(L)
        tens = make_splitted_mps(tens)
        target_mps = qtn.MatrixProductState(tens, shape='lrp')
        target_mps.normalize()
        

    if mps_type == 'random':
        L = 16
        target_mps = qtn.MPS_rand_state(L=L, bond_dim=4)
        target_mps.permute_arrays(shape='lrp')
        
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
        
    
    mps_p = MPSPreparation(target_mps, qubit_hamiltonian=qubit_hamiltonian)
    # # 
    # number_of_layers = 4
    # mpsp.seq_preparation(number_of_layers, do_compression=False, max_bond_dim=64, verbose=True)
    
    #
    # number_of_layers = 2
    # mps_p.variational_seq_preparation(number_of_layers, do_compression=False, max_bond_dim=64, verbose=True)
    
    # # 
    depth = 8
    mps_p.qctn_preparation(depth)
    
    # number_of_lcu_layers = 4  
    # mpsp.lcu_preparation(number_of_lcu_layers, verbose=False)
    
    # number_of_lcu_layers = 4
    # mpsp.variational_lcu_preparation(number_of_lcu_layers, verbose=False)
    

    #### 1d adiabatic state preparation - random D=d=2 mps
    # mps_p = MPSPreparation(target_mps)
    # Tmax, tau = 32, 0.04 #total runtime, trotter step size
    # max_bond = 2
    # mps_p.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)
    
    # plt.plot(mps_p.adiabatic_data['target_fidelity'].keys(), 
    #          mps_p.adiabatic_data['target_fidelity'].values(), '.-')

    
    #### 1d adiabatic state preparation - aklt
    # L=8
    # tensor_array, _ = make_aklt_mps(L)
    # target_mps = qtn.MatrixProductState(tensor_array, shape='lrp')
    # mps_p = MPSPreparation(target_mps)
    
    # Tmax, tau = 6, 0.04 #total runtime, trotter step size
    # max_bond = 2
    # mps_p.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)
    
    # plt.plot(mps_p.adiabatic_data['target_fidelity'].keys(), 
    #          mps_p.adiabatic_data['target_fidelity'].values(), '.-')
    
    
    #### 2d adiabatic state preparation    
    # Lx, Ly = 10, 2
    # target_grid, _ = make_aklt_peps(Lx, Ly)
    # peps_p = PEPSPreparation(target_grid)
    
    # Tmax, tau = 6, 0.04
    # max_bond = 2
    # peps_p.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)

    # plt.plot(peps_p.adiabatic_data['target_fidelity'].keys(), 
    #           peps_p.adiabatic_data['target_fidelity'].values(), '.-')
    
    
