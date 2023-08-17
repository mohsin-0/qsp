#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pytket.circuit import Circuit, Unitary1qBox, Unitary2qBox
from pytket.passes import DecomposeBoxes
from pytket.extensions.qiskit import AerBackend
import quimb.tensor as qtn

import tsp_helper_routines as tsp_hr

AER_BACKEND = AerBackend()


def overlap_value(al, psi, approx_psi, residual, qubit_hamiltonian=0):
    return -np.abs(tsp_hr.norm_mps_ovrlap(psi, approx_psi + al[0]*residual))
    

def apply_quimb_unitary(u, quimb_circ, it, it_plus_1):
    if u.shape[0]==4:
        circ = Circuit(2).add_unitary2qbox(Unitary2qBox(u), 0,1)
        DecomposeBoxes().apply(circ)
        compiled_circ = AER_BACKEND.get_compiled_circuit(circ, optimisation_level=1)
    
    else:
        circ = Circuit(1).add_unitary1qbox(Unitary1qBox(u), 0)
        DecomposeBoxes().apply(circ)
        compiled_circ = AER_BACKEND.get_compiled_circuit(circ, optimisation_level=1)
        
    for command in compiled_circ.get_commands():
        
        if command.op.type.name=='TK1':
            qubit_id = (it, it_plus_1)[command.qubits[0].index[0]]
            params = command.op.params
        
            quimb_circ.apply_gate('RZ', params[2]*np.pi, qubit_id, parametrize=True)
            quimb_circ.apply_gate('RX', params[1]*np.pi, qubit_id, parametrize=True)
            quimb_circ.apply_gate('RZ', params[0]*np.pi, qubit_id, parametrize=True)
            
        elif command.op.type.name=='CX': 
            quimb_circ.apply_gate('CX', it, it_plus_1)
            
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
    return -abs((circ_unitary.H & target_mps & zero_wfn).contract(all, 
                                                                  optimize='auto-hq', 
                                                                  backend='tensorflow'))

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
        tags=['RX','RZ'],                           # only optimize RX and RZ tensors
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

                            