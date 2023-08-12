#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt

import quimb.tensor as qtn

from tsp import apply_unitary_layers_on_wfn
from tsp_sequ import generate_sequ_for_mps
from tsp_lcu import generate_lcu_for_mps
from tsp_sequ_wt_autodiff_parametrized import sequ_unitary_circuit_optimization
from tsp_lcu_wt_autodiff_remannian import lcu_unitary_circuit_optimization

from utils import (norm_mps_ovrlap, cl_zero_mps, compute_energy_expval)



if __name__ == "__main__":
    
    # DONE: Decompose into CNOT + OQGs and optimize OQG with autodiff gradient.
    # TODO: Optimize two-qubit gates over isometric manifold.
    # TODO: QCTN (Quantum Circuit Tensor Networks (also works in 2D))
    # TODO: isoPEPS preparation not published yet.
    # TODO: MPS in Quimb Format -> Gates? (also for LCU circuits)    
    # TODO: Input and format of the MPS - should be framework agnostic
    # TODO: LCU to tket circuit
    # TODO: Break 4 qubit gate
    # TODO: 2d AKLT
    
    overlaps_gap = 4

    qubit_hamiltonian = 0
    preparation_method = 'SeqU'#'LCU'#
    mps_type = 'aklt'#''random'#'random'#'P4'#'heisenberg'#'N2'#
    
    data_dict = {}
    if mps_type == 'aklt':
        
        from adiabatic_time_evolution import make_1d_aklt_tensor
        Q_aklt = make_1d_aklt_tensor()

        ten = qtn.Tensor(Q_aklt.reshape((2,2, 2,2)), inds=('vl','vr','pl','pr') )
        tn = ten.split(('vl','pl'), bond_ind='v0')
        ten0, ten1 = tn.tensor_map[0], tn.tensor_map[1]
        ten0 = (tn.tensor_map[0]).transpose(*('vl', 'v0', 'pl'))
        ten1 = (tn.tensor_map[1]).transpose(*('v0', 'vr', 'pr'))
        
        ten0 = np.array(ten0.data, dtype=np.float64)
        ten1 = np.array(ten1.data, dtype=np.float64)
        
        L=16
        cyclic = False
        Qs = []
        for _ in range(L):
            Qs.append(ten0)
            Qs.append(ten1)
        
        if not cyclic:
            Qs[ 0] = np.squeeze(Qs[ 0][0, :, :])
            Qs[-1] = np.squeeze(Qs[-1][:, 0, :])
        mps = qtn.MatrixProductState(Qs, shape='lrp')
        
        params = [0]
        data = {}
        data['quimb_mps'] = mps
        data['qubit_hamiltonian'] = 0
        data_dict[0] = data
        
    
    if mps_type == 'random':
        params = [0]
        data = {}
        data['quimb_mps'] = (qtn.MPS_rand_state(L=16, bond_dim=4) + 
                             qtn.MPS_rand_state(L=16, bond_dim=4)*0*1j)
        
        data['qubit_hamiltonian'] = 0
        data_dict[0] = data
        
    
    if mps_type == 'P4':
        params = [1.9, 2, 2.1]
        for param in params:
            filename = f'/home/mohsin/Documents/cqc_github/TSP/data_tsp/P4_6-31G/dist{param:0.4f}.pkl'
            with open(filename, 'rb') as f:
                data_dict[param] = pkl.load(f)
                
    if mps_type == 'N2':
        params = [1.8, 2.0, 2.2]
        for param in params:
            filename = f'/home/mohsin/Documents/cqc_github/TSP/data_tsp/N2_STO-6G/dist{param:0.4f}.pkl'
            with open(filename, 'rb') as f:
                data_dict[param] = pkl.load(f)
    
    if mps_type == 'heisenberg':
        params = [0.6,0.8,1]
        for param in params:
            filename = f'/home/mohsin/Documents/cqc_github/TSP/data_tsp/heisenberg_L32/dist{param:0.4f}.pkl'
            with open(filename, 'rb') as f:
                data_dict[param] = pkl.load(f)
    
    
    data = data_dict[params[0]]
    target_mps = data['quimb_mps']
    qubit_hamiltonian = data['qubit_hamiltonian']
    target_mps.permute_arrays(shape='lpr')
    target_mps.compress('right')
    L = target_mps.L
    
    
    for preparation_method in ['LCU+autodiff']:#['SeqU']:#['SeqU+autodiff']:#['LCU+autodiff']:#['SeqU', 'SeqU+autodiff', 'LCU', 'LCU+autodiff']:
        
        if preparation_method=='SeqU':
            italic_D_sequ = 24
            preparation_data = generate_sequ_for_mps(target_mps, qubit_hamiltonian, italic_D_sequ, do_compression=True, max_bond_dim=64, verbose=False)
            unitaries = preparation_data['unitaries']
            
            encoded_mps = apply_unitary_layers_on_wfn(unitaries, cl_zero_mps(L))
            encoded_mps.right_canonize(normalize=True)
            quimb_overlap = norm_mps_ovrlap(encoded_mps, target_mps)
            print('SeqU - overlaps:',  preparation_data['overlaps'][-1], quimb_overlap - preparation_data['overlaps'][-1] )


        if preparation_method=='SeqU+autodiff':
            italic_D_sequ = 4
            preparation_data = generate_sequ_for_mps(target_mps, qubit_hamiltonian, italic_D_sequ, do_compression=False, verbose=False)
            unitaries = preparation_data['unitaries']
            
            encoded_mps = apply_unitary_layers_on_wfn(unitaries, cl_zero_mps(L))
            encoded_mps.right_canonize(normalize=True)
            quimb_overlap = norm_mps_ovrlap(encoded_mps, target_mps)
            print('SeqU - overlaps:',  preparation_data['overlaps'][-1], quimb_overlap - preparation_data['overlaps'][-1] )
            
            # from untitled1 import sequ_unitary_circuit_optimization 
            # sequ_unitary_circuit_optimization(target_mps, unitaries)
            optimized_unitary = sequ_unitary_circuit_optimization(target_mps, unitaries)
            

        if preparation_method=='LCU':
            italic_D_lcu = 24
            preparation_data = generate_lcu_for_mps( target_mps, qubit_hamiltonian, italic_D_lcu, do_compression=True, verbose=False)        
            
            kappas, unitaries = preparation_data['kappas'], preparation_data['unitaries']
            
            zero_wfn = cl_zero_mps(L)
            lcu_mps = [apply_unitary_layers_on_wfn(curr_us, zero_wfn) for curr_us in unitaries]
            
            encoded_mps = cl_zero_mps(L)*0
            for kappa, curr_mps in zip(kappas, lcu_mps):
                encoded_mps = encoded_mps + kappa*curr_mps
            encoded_mps.right_canonize(normalize=True)
            quimb_overlap = norm_mps_ovrlap(encoded_mps, target_mps)
            print('LCU - overlap:', preparation_data['overlaps'][-1],  quimb_overlap - preparation_data['overlaps'][-1] )
            
            
        if preparation_method=='LCU+autodiff':
            italic_D_lcu = 4
            preparation_data = generate_lcu_for_mps( target_mps, qubit_hamiltonian, italic_D_lcu, do_compression=True, verbose=False)        
            
            kappas, unitaries = preparation_data['kappas'], preparation_data['unitaries']
            
            zero_wfn = cl_zero_mps(L)
            lcu_mps = [apply_unitary_layers_on_wfn(curr_us, zero_wfn) for curr_us in unitaries]
            
            encoded_mps = cl_zero_mps(L)*0
            for kappa, curr_mps in zip(kappas, lcu_mps):
                encoded_mps = encoded_mps + kappa*curr_mps
            encoded_mps.right_canonize(normalize=True)
            quimb_overlap = norm_mps_ovrlap(encoded_mps, target_mps)
            print('LCU - overlap:', preparation_data['overlaps'][-1],  quimb_overlap - preparation_data['overlaps'][-1] )
            
            # lcu_unitary_circuit_optimization(target_mps, kappas, lcu_mps)
            
            pkl.dump([target_mps, kappas, lcu_mps], open('temp_dump.pkl', "wb"))
            
            # plt.plot([1+overlap.numpy() for overlap in overlaps_list], '.-b')
            # plt.yscale('log')
            # plt.xlabel('iter')
            # plt.ylabel('overlaps')
            # plt.tight_layout()
            
            
        # energy_dmrg = data['energy_dmrg']
        # it_Ds = preparation_data['it_Ds']
        # overlaps = preparation_data['overlaps']
        # energies = preparation_data['energies']
        
        # x = it_Ds
        # y1 = 1-np.array(overlaps)
        # # y2 = ent_entropies  
        # y3 = np.real((np.array(energies)-energy_dmrg)*1000)
        
        # plt.subplot(1,2,1)
        # plt.title('overlaps')
        # plt.plot(x, y1, '.-', label=fr'${preparation_method}, dist={param}$')
        # plt.yscale('log')
        # plt.xlabel(r'$No.\ of\ unitary\ approximation\ layers,\ \mathcal{D}$')
        # plt.ylabel(r'$-\log \left(Unitary\ approx.\ overlap\ with\ the\ givan\ mps \right)$')
        # plt.legend()
        # # labellines.labelLines(plt.gca().get_lines())
        
        # plt.subplot(1,2,2)
        # plt.title('energy error')
        # # if prob.description == "heisenberg" or prob.description == "ising":
        #     # y3 = y3/1000
        # plt.plot(x, y3, '.-', label=fr'${preparation_method}, dist={param}$')
        # plt.xlabel(r'$No.\ of\ unitary\ approximation\ layers,\ \mathcal{D}$')
        # plt.ylabel(r'$err. = E_{approx.}-E_{DMRG}\ (mH)$')
        # plt.yscale('log')
        
        # plt.plot(x, np.real(x)*0+1.58, 'k-')
    # plt.tight_layout()