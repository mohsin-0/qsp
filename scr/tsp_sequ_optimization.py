#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import qiskit 
import quimb.tensor as qtn

import tsp_helper_routines as tsp_hr


def overlap_value(al, psi, approx_psi, residual, qubit_hamiltonian=0):
    return -np.abs(tsp_hr.norm_mps_ovrlap(psi, approx_psi + al[0]*residual))
    

def apply_quimb_unitary(u, quimb_circ, it, it_plus_1):
    n_qubits = int(np.log2(u.shape[0]))
    
    circ = qiskit.QuantumCircuit(n_qubits)
    circ.unitary(u, list(range(n_qubits)))

    if n_qubits==1:   
        # single qubit unitary, to take care of the reverse ordering in the single-case
        it_plus_1 = it
        
    trans_qc = qiskit.transpile(circ, basis_gates=['cx','u3'])
    for instruction in trans_qc:
        if instruction.operation.name=='u3':
            qubit_id = (it_plus_1,it)[instruction.qubits[0].index]
            params = instruction.operation.params
            quimb_circ.apply_gate('U3', params[0], params[1], params[2], qubit_id, parametrize=True)
                
        elif instruction.operation.name=='cx':
            quimb_circ.apply_gate('CX', it_plus_1, it)
             
        else:
            print('unhandled gate')
            exit()
        
            
def generate_circ_from_unitary_layers(unitary_layers, L):

    quimb_circ = qtn.Circuit(L)
    for it in reversed(range(len(unitary_layers))):
        Gs_lst = unitary_layers[it]
        for start_indx, end_indx, Gs, _, _ in Gs_lst:
            for it in range(start_indx, end_indx+1):
                if (it==end_indx):
                    apply_quimb_unitary(Gs[it-start_indx].data, quimb_circ, it, it+1)
    
                else:                
                    apply_quimb_unitary(Gs[it-start_indx].data, quimb_circ, it, it+1)
                        
    return quimb_circ


################################################
def loss(circ_unitary, zero_wfn, target_mps):
    """returns -abs(<target_mps|circ_unitary|zero_wfn>)
        assumes that target_mps and zero_wfn are normalized
    """
    return -abs((circ_unitary.H & target_mps & zero_wfn).
                contract(all, optimize='auto-hq', backend='tensorflow'))


def sequ_unitary_circuit_optimization(target_mps, unitaries): 
    """build parametrized circuit from sequential unitary ansatz
    """    
    quimb_circ = generate_circ_from_unitary_layers(unitaries, target_mps.L)
        
    indx_map = {}
    for it in range(target_mps.L):
        indx_map[f'k{it}']=f'b{it}'
    target_mps = target_mps.reindex(indx_map)
    
    circ_unitary = quimb_circ.uni
    print(f'number of gates in the circuit (from sequential algorithm) are : {quimb_circ.num_gates}') 
    
    zero_wfn = tsp_hr.cl_zero_mps(target_mps.L)
    zero_wfn = zero_wfn.astype(np.complex64)
    target_mps = target_mps.astype(np.complex64)
    circ_unitary = circ_unitary.astype(np.complex64)

    print(f'overlap before optimization = {loss(circ_unitary, zero_wfn, target_mps):.10f}\n')
    tnopt = qtn.TNOptimizer(
        circ_unitary,                               # tensor network to optimize
        loss,                                       # function to minimize
        loss_constants={'zero_wfn':zero_wfn,        # static inputs
                        'target_mps': target_mps},  
        tags=['U3'],                                
        autodiff_backend='tensorflow', optimizer='L-BFGS-B'
    )
    unitary_opt = tnopt.optimize_basinhopping(n=1000, nhop=4)
    return unitary_opt


if __name__ == "__main__":
    pass    
    # mps_type = 'random'#'molecule'#
    # L = 16
    # italic_D_sequ = 4
    # with open(f'data_preparations/sequ_{mps_type}_L{L}_layers{italic_D_sequ}.pkl', 'rb') as f:
    #     [target_mps, unitaries, quimb_overlap] = pkl.load(f)
    # print('quimb_overlap: ', quimb_overlap)                
    # unitary_circuit_optimization(target_mps, unitaries)

                            