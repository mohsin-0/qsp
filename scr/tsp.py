#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import quimb.tensor as qtn

import tsp_unitary as tspu
import tsp_helper_routines as tsp_hr

from tsp_sequ_optimization import sequ_unitary_circuit_optimization
from tsp_lcu_optimization import lcu_unitary_circuit_optimization
from tsp_qctn import quantum_circuit_tensor_network_ansatz

from tsp_adiabatic_1d import adiabatic_state_preparation_1d
from tsp_misc_tns import make_bell_pair_mps

class PEPSPreparation():
    pass


class MPSPreparation():
    def __init__(self, tensor_array, shape='lrp', qubit_hamiltonian=0):
        
        if isinstance(tensor_array, qtn.tensor_1d.MatrixProductState):
            target_mps = tensor_array        
        else:
            target_mps = qtn.MatrixProductState(tensor_array, shape=shape)
        
        self.target_mps = target_mps
        self.shape = target_mps.shape
        self.L = target_mps.L
        self.qubit_hamiltonian = qubit_hamiltonian
        
        
    def seq_preparation(self, number_of_layers, 
                        do_compression=False, max_bond_dim=64, 
                        verbose=True):
        
        data = tspu.generate_sequ_for_mps(self.target_mps, number_of_layers, 
                                          do_compression=do_compression, 
                                          max_bond_dim=max_bond_dim, 
                                          verbose=verbose, 
                                          qubit_hamiltonian=self.qubit_hamiltonian)
        self.seq_preparation_data = data
        
        unitaries = data['unitaries']
        encoded_mps = tspu.apply_unitary_layers_on_wfn(unitaries, tsp_hr.cl_zero_mps(self.L))
        encoded_mps.right_canonize(normalize=True)
        overlap = tsp_hr.norm_mps_ovrlap(encoded_mps, self.target_mps)
        
        assert np.abs(overlap-data['overlaps'][-1]) < 1e-14, 'overlap from seq unitary does not match!'
        print('final overlap from seq. preparation:',  overlap)
        
        
    def variational_seq_preparation(self, number_of_layers, 
                                    do_compression=False, max_bond_dim=64, 
                                    verbose=True):
        
        self.seq_preparation(number_of_layers, 
                            do_compression=do_compression, max_bond_dim=max_bond_dim, 
                            verbose=verbose)

        unitaries = self.seq_preparation_data['unitaries']
        optimized_unitary = sequ_unitary_circuit_optimization(self.target_mps, unitaries)
        

    def lcu_preparation(self, number_of_lcu_layers, verbose=False):
        data = tspu.generate_lcu_for_mps(self.target_mps, 
                                         number_of_lcu_layers,
                                         qubit_hamiltonian=self.qubit_hamiltonian, 
                                         verbose=verbose)
        
        self.lcu_preparation_data = data
        
        kappas, unitaries = data['kappas'], data['unitaries']
        
        zero_wfn = tsp_hr.cl_zero_mps(self.L)
        lcu_mps = [tspu.apply_unitary_layers_on_wfn(curr_us, zero_wfn) for curr_us in unitaries]
        
        encoded_mps = tsp_hr.cl_zero_mps(self.L)*0
        for kappa, curr_mps in zip(kappas, lcu_mps):
            encoded_mps = encoded_mps + kappa*curr_mps
        encoded_mps.right_canonize(normalize=True)
        overlap = tsp_hr.norm_mps_ovrlap(encoded_mps, self.target_mps)
        
        assert np.abs(overlap-data['overlaps'][-1]) < 1e-14, 'overlap from lcu unitary does not match!'
        print('final overlap from seq. preparation:',  overlap)
        
        
    def variational_lcu_preparation(self, number_of_lcu_layers, verbose=False):
        self.lcu_preparation(number_of_lcu_layers, verbose=verbose)
        
        data = self.lcu_preparation_data
        kappas, unitaries = data['kappas'], data['unitaries']
        zero_wfn = tsp_hr.cl_zero_mps(self.L)
        lcu_mps = [tspu.apply_unitary_layers_on_wfn(curr_us, zero_wfn) for curr_us in unitaries]
        
        lcu_unitary_circuit_optimization(self.target_mps, kappas, lcu_mps)
            
        
    def qctn_preparation(self, depth):
        quantum_circuit_tensor_network_ansatz(self.target_mps, depth)
    
            
    def adiabatic_state_preparation(self):
        L = 8
        Tmax, tau = 6, 0.04 #total runtime, trotter step size
        max_bond = 2
        
        s_func = lambda t: np.sin( (np.pi/2)*np.sin( (np.pi/2)*t/Tmax )**2 )**2
        # s_func = lambda t: np.sin( (np.pi/2)*t/Tmax)**2
        # s_func = lambda t: t/Tmax
        
        # # ####################################        
        initial_tens   = make_bell_pair_mps(L=L)
        initial_mps = qtn.MatrixProductState(initial_tens, shape='lrp')
        
        self.adiabatic_data = adiabatic_state_preparation_1d(self.target_mps, 
                                                             initial_mps, 
                                                             Tmax, tau, s_func, 
                                                             max_bond)

        