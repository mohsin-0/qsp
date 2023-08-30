#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import quimb.tensor as qtn

import tsp_unitary as tspu
import tsp_helper_routines as tsp_hr

from tsp_lcu_optimization_manopt import lcu_unitary_circuit_optimization as lcu_manopt
from tsp_lcu_optimization_qgopt  import lcu_unitary_circuit_optimization as lcu_qgopt

from tsp_sequ_optimization import seq_unitary_circuit_opti


from tsp_qctn import quantum_circuit_tensor_network_ansatz

from tsp_adiabatic_1d import adiabatic_state_preparation_1d
from tsp_adiabatic_2d import adiabatic_state_preparation_2d

from tsp_misc_tns import make_bell_pair_mps
from tsp_misc_tns import make_bell_peps
        
            
class MPSPreparation():
    def __init__(self, 
                 tensor_array, 
                 shape='lrp'):
        
        if isinstance(tensor_array, qtn.MatrixProductState):
            target_mps = tensor_array    
            
        else:
            target_mps = qtn.MatrixProductState(tensor_array, shape=shape)
        
        self.target_mps = target_mps
        self.shape = target_mps.shape
        self.L = target_mps.L
        
        
    def seq_preparation(self, 
                        number_of_layers, 
                        do_compression=False, 
                        max_bond_dim=64, 
                        verbose=False):
        
        data = tspu.generate_sequ_for_mps(self.target_mps, 
                                          number_of_layers, 
                                          do_compression=do_compression, 
                                          max_bond_dim=max_bond_dim, 
                                          verbose=verbose)
        
        self.seq_data = data
        unitaries, circ = data['unitaries'], data['circ']
        
        # sanity check
        encoded_mps = tspu.apply_unitary_layers_on_wfn(unitaries, 
                                                       tsp_hr.cl_zero_mps(self.L))
        
        encoded_mps.right_canonize(normalize=True)
        overlap = tsp_hr.norm_mps_ovrlap(encoded_mps, self.target_mps)    
        assert (np.abs(overlap-data['overlaps'][-1]) < 1e-14, 
                'overlap from seq unitary does not match!')

        print(f'overlap from static seq. preparation = {np.abs(overlap):0.8f}, '
              f'n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}')

        
    def variational_seq_preparation(self, 
                                    number_of_layers, 
                                    n_iter, 
                                    nhop,
                                    do_compression=False, 
                                    max_bond_dim=64, 
                                    verbose=False):
        
        if not hasattr(self, 'seq_data'):
            self.seq_preparation(number_of_layers, 
                                 do_compression=do_compression, 
                                 max_bond_dim=max_bond_dim, 
                                 verbose=False)
        
        print('\nnow doing variational optimization over gates')
        self.var_seq_data = seq_unitary_circuit_opti(self.target_mps, 
                                                     self.seq_data['unitaries'],
                                                     n_iter, 
                                                     nhop)
        
        circ, tnopt = self.var_seq_data['circ'], self.var_seq_data['tnopt']
        
        print(f'overllap after variational optimization = {-tnopt.loss_best:0.8f}, ',
              f'n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}')
        
 
    def qctn_preparation(self, depth, n_iter, nhop):
        self.qctn_data = quantum_circuit_tensor_network_ansatz(self.target_mps, 
                                                               depth, 
                                                               n_iter, 
                                                               nhop,)
        
        circ, tnopt = self.qctn_data['circ'], self.qctn_data['tnopt']
        print(f'overllap after qctn optimization = {-tnopt.loss_best:0.8f}, ',
              f'n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}')
    
        
    def lcu_preparation(self, 
                        number_of_lcu_layers, 
                        verbose=False):
        
        data = tspu.generate_lcu_for_mps(self.target_mps, 
                                         number_of_lcu_layers,
                                         verbose=verbose)
        
        self.lcu_data = data
        kappas, unitaries = data['kappas'], data['unitaries']
        
        zero_wfn = tsp_hr.cl_zero_mps(self.L)
        lcu_mps = [tspu.apply_unitary_layers_on_wfn(curr_us, zero_wfn) 
                   for curr_us in unitaries]
        
        encoded_mps = tsp_hr.cl_zero_mps(self.L)*0
        for kappa, curr_mps in zip(kappas, lcu_mps):
            encoded_mps = encoded_mps + kappa*curr_mps
        encoded_mps.right_canonize(normalize=True)
        overlap = tsp_hr.norm_mps_ovrlap(encoded_mps, self.target_mps)
        
        assert (np.abs(overlap-data['overlaps'][-1]) < 1e-14, 
                f"overlap from lcu unitary does not match! {overlap}!={data['overlaps'][-1]}")
        
        kappas_temp = np.zeros(len(kappas))
        kappas_temp[0] = 1.

        import qiskit
        from lcu_circuit import apply_lcu_with_layers
        
        k = int(np.ceil(np.log2(len(kappas))))
        L = self.L
        circ = qiskit.QuantumCircuit(L+k+1)
        circ, overlap_from_lcu_circ = apply_lcu_with_layers(circ, 
                                                            kappas, 
                                                            unitaries, 
                                                            self.target_mps)
        circ = qiskit.transpile(circ, basis_gates=['cx','u3'])
        
        print('overlap to target mps from constructed lcu circuit =', 
               overlap_from_lcu_circ)

        print(f'overllap after lcu. preparation = {np.abs(overlap):.8f}, ',
               f'n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}')
        
        
    def variational_lcu_preparation(self, 
                                    number_of_lcu_layers, 
                                    max_iterations,
                                    verbose=False):
        
        if not hasattr(self, 'lcu_data'):
            self.lcu_preparation(number_of_lcu_layers, verbose=False)
        
        kappas, unitaries = self.lcu_data['kappas'], self.lcu_data['unitaries']
        lcu_mps = [tspu.apply_unitary_layers_on_wfn(curr_us, 
                                                    tsp_hr.cl_zero_mps(self.L)) 
                   for curr_us in unitaries]
       
        print('\nnow doing variational optimization over LCU ansatz')
        method_name = ''
        if all([D==2 for mps in lcu_mps for D in mps.bond_sizes()]):
            method_name = 'manopt'
            self.manopt_data = lcu_manopt(self.target_mps, 
                                          kappas, 
                                          lcu_mps, 
                                          max_iterations=max_iterations, 
                                          verbose=verbose)
            
            lcu_mps_opt = self.manopt_data['lcu_mps_opt']
            kappas = self.lcu_data['kappas']

        else:
            method_name = 'qgopt'
            self.qgopt_data = lcu_qgopt(self.target_mps, 
                                        kappas, 
                                        lcu_mps,
                                        max_iterations=max_iterations,
                                        verbose=verbose)  
            
            lcu_mps_opt = self.qgopt_data['lcu_mps_opt']
            kappas = self.lcu_data['kappas']
            
        encoded_mps = tsp_hr.cl_zero_mps(self.L)*0
        for kappa, curr_mps in zip(kappas, lcu_mps_opt):
            encoded_mps = encoded_mps + kappa*curr_mps
        encoded_mps.right_canonize(normalize=True)
        overlap = tsp_hr.norm_mps_ovrlap(encoded_mps, self.target_mps)
        print(f'overllap after lcu optimization ({method_name}) = {np.abs(overlap):.8f}\n')
        
            
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




class PEPSPreparation():
    def __init__(self, tensor_grid, shape='ldrup'):
        self.target_grid = tensor_grid
        self.shape = shape
        
        
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
        