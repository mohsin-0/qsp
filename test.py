#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl

import quimb.tensor as qtn

from tsp import apply_unitary_layers_on_wfn
from tsp_sequ import generate_sequ_for_mps
from tsp_lcu import generate_lcu_for_mps
from tsp_sequ_wt_autodiff_parametrized import sequ_unitary_circuit_optimization
from tsp_lcu_wt_autodiff_remannian import lcu_unitary_circuit_optimization

from utils import (norm_mps_ovrlap, cl_zero_mps, compute_energy_expval)
from make_aklt import make_aklt_1d_mps


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
    mps_type = 'random'#'random'#'P4'#'heisenberg'#'N2'#'aklt'#
    
    data_dict = {}
    if mps_type == 'aklt':
        L = 8
        mps, _ = make_aklt_1d_mps(L=L)
        mps = qtn.MatrixProductState(mps, shape='lrp')
        
        data = {}
        data['quimb_mps'] = mps
        data['qubit_hamiltonian'] = 0
        
        
    if mps_type == 'random':
        data = {}
        data['quimb_mps'] = (qtn.MPS_rand_state(L=16, bond_dim=4) + 
                             qtn.MPS_rand_state(L=16, bond_dim=4)*0*1j)
        data['qubit_hamiltonian'] = 0.
        
    
    if mps_type in ['P4','N2','heisenberg']:
        filenames = {'P4': 'test_data/P4_6-31G_dist2.0000.pkl', 
                     'N2': 'test_data/N2_STO-6G_dist2.0000.pkl',
                     'heisenberg':'test_data/heisenberg_L32_dist0.8000.pkl'}
        
        with open(filenames[mps_type], 'rb') as f:
            data = pkl.load(f)
        
        
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