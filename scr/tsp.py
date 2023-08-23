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
from tsp_adiabatic_2d import adiabatic_state_preparation_2d

from tsp_misc_tns import make_bell_pair_mps
from tsp_misc_tns import make_bell_peps


class PEPSPreparation():
    def __init__(self, tensor_grid, shape='ldrup', qubit_hamiltonian=0):
        self.target_grid = tensor_grid
        self.shape = shape
        self.qubit_hamiltonian = qubit_hamiltonian
        
        self.Lx, self.Ly = len(tensor_grid[0]), len(tensor_grid)
        self.phy_dim = tensor_grid[0][0].shape[-1]
            
        
    def adiabatic_state_preparation(self, Tmax, tau, max_bond, verbose=False):        
        # s_func = lambda t: np.sin( (np.pi/2)*np.sin( (np.pi/2)*t/Tmax )**2 )**2
        s_func = lambda t: np.sin( (np.pi/2)*t/Tmax)**2
        # s_func = lambda t: t/Tmax
        
        # target_grid, bonds = make_aklt_peps(Lx, Ly)
        initial_grid, bonds = make_bell_peps(self.Lx, self.Ly)
        
        data = adiabatic_state_preparation_2d(self.target_grid, 
                                       initial_grid, bonds, 
                                       self.Lx, self.Ly, self.phy_dim, 
                                       Tmax, tau, max_bond, s_func, 
                                       verbose=verbose)
        self.adiabatic_data = data
        
        t_last = max(data['ss'].keys())
        s, e, f = data['ss'][t_last], data['energy'][t_last], data['target_fidelity'][t_last]
        print(f"\n2d adiabatic preparation: @ {s=:.5f}, e={e:.08f} and f={f:.08f}\n")
        
            
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
    
            
    def adiabatic_state_preparation(self, Tmax, tau, max_bond, verbose=False):
        
        print('adiabatic state preparation of mps:\n'
              f'runtime={Tmax}, tau={tau:0.04}, steps={int(Tmax/tau)}, max_bond={max_bond}\n')
                
        Ds = self.target_mps.bond_sizes()
        D, d = max(Ds), self.target_mps.phys_dim()
        
        # assumes uniform bond dimension
        assert all(i == Ds[0] for i in Ds)
        
        if D**2 > d:
            print('given mps is not injective. blocking it now ...')
            # block the to be in injective form
            block_size = int(np.ceil(2*np.log(D)/np.log(d)))
            blocked_mps = tsp_hr.blockup_mps(self.target_mps, block_size)
        else:
            blocked_mps = self.target_mps
            
            
        s_func = lambda t: np.sin( (np.pi/2)*np.sin( (np.pi/2)*t/Tmax )**2 )**2
        # s_func = lambda t: np.sin( (np.pi/2)*t/Tmax)**2
        # s_func = lambda t: t/Tmax
        
        # # ####################################        
        initial_tens   = make_bell_pair_mps(L=blocked_mps.L, phys_dim=blocked_mps.phys_dim())
        initial_mps = qtn.MatrixProductState(initial_tens, shape='lrp')
        
        data = adiabatic_state_preparation_1d(blocked_mps, initial_mps, 
                                              Tmax, tau, s_func, max_bond, verbose=verbose)
        
        self.adiabatic_data = data
        t_last = max(data['ss'].keys())    
        s, e = data['ss'][t_last], data['energy'][t_last]
        curr_f, tar_f = data['current_fidelity'][t_last], data['target_fidelity'][t_last],
        print(f"final overlap @ {s=:.5f} is e={e:.08f}, "
              f"curr_f={curr_f:.08f}, and target_fid={tar_f:.08f}\n")
        