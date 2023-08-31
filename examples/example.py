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
    # TODO: Input and format of the MPS - should be framework agnostic - tensor array with shape
    # TODO: sanity check for TSP - d==chi
    # TODO: add references + add documentation + comments
    # TODO: LCU to qiskit circuit - phase problem
    # TODO: qiskit circut for adiabatic 1d, abiabatic 2d
    # TODO: Notebooks - one example notebook. one notebook for benchmarks
    # TODO: pip
    # TODO: fix verbose
    # TODO ordering of input arguments and type checking
    # TODO some costing for lcu + adiabatic
    
    mps_type = 'N2'#'aklt'#'heisenberg'#'aklt'#
    
    if mps_type == 'aklt':
        L = 8
        tens, bond = make_aklt_mps(L)
        tens = make_splitted_mps(tens)
        target_mps = qtn.MatrixProductState(tens, shape='lrp')
        target_mps.normalize()
        
        
    if mps_type == 'random':
        L = 12
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
        

    mps_p = MPSPreparation(target_mps)
    
    # #
    # number_of_layers = 4
    # mps_p.seq_preparation(number_of_layers, do_compression=False, verbose=False)
    # print('')
    
    # #
    # number_of_layers = 4
    # n_iter, nhop = 40, 4
    # mps_p.variational_seq_preparation(number_of_layers, 
    #                                   do_compression=False, 
    #                                   n_iter=n_iter, 
    #                                   nhop=nhop,
    #                                   verbose=False)
    # print('\n\n')
    
    # # 
    # depth = 8
    # n_iter, nhop = 10, 4,
    # mps_p.qctn_preparation(depth, n_iter=n_iter, nhop=nhop)
    # print('\n\n')
    
    #
    # number_of_lcu_layers = 4
    # mps_p.lcu_preparation(number_of_lcu_layers, verbose=False)
        
    #
    number_of_lcu_layers = 4
    max_iterations = 31
    mps_p.variational_lcu_preparation(number_of_lcu_layers, 
                                      max_iterations, 
                                      verbose=False)
    # print('\n\n')
    
    
    #### 1d adiabatic state preparation - random D=d=2 mps
    # mps_p = MPSPreparation(target_mps)
    # Tmax, tau = 32, 0.04 #total runtime, trotter step size
    # max_bond = 2
    # mps_p.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)
    
    # plt.plot(mps_p.adiabatic_data['target_fidelity'].keys(), 
    #          mps_p.adiabatic_data['target_fidelity'].values(), '.-')

    
    #### 1d adiabatic state preparation - aklt
    # tensor_array, _ = make_aklt_mps(L)
    # target_mps = qtn.MatrixProductState(tensor_array, shape='lrp')
    # mps_p = MPSPreparation(target_mps)
    
    # Tmax, tau = 6, 0.04 #total runtime, trotter step size
    # max_bond = 2
    # mps_p.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)
    
    # plt.plot(mps_p.adiabatic_data['target_fidelity'].keys(), 
    #           mps_p.adiabatic_data['target_fidelity'].values(), '.-')
    
    
    #### 2d adiabatic state preparation    
    # Lx, Ly = 10, 2
    # target_grid, _ = make_aklt_peps(Lx, Ly)
    # peps_p = PEPSPreparation(target_grid)
    
    # Tmax, tau = 6, 0.04
    # max_bond = 2
    # peps_p.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)

    # plt.plot(peps_p.adiabatic_data['target_fidelity'].keys(), 
    #           peps_p.adiabatic_data['target_fidelity'].values(), '.-')
