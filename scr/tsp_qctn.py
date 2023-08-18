#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pytket.circuit import Circuit, Unitary1qBox, Unitary2qBox
from pytket.passes import DecomposeBoxes
from pytket.extensions.qiskit import AerBackend

import quimb as qu
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


# quantum circuit tensor network ansatz
def single_qubit_layer(circ, gate_round=None):
    """Apply a parametrizable layer of single qubit ``U3`` gates.
    """
    for i in range(circ.N):
        # initialize with random parameters
        
        params = qu.randn(3, dist='uniform')
        # circ.apply_gate('U3', *params, i, gate_round=gate_round, parametrize=True)
        circ.apply_gate('RZ', params[0], i, gate_round=gate_round, parametrize=True)
        circ.apply_gate('RX', params[1], i, gate_round=gate_round, parametrize=True)
        circ.apply_gate('RZ', params[2], i, gate_round=gate_round, parametrize=True)
        
        
def two_qubit_layer(circ, gate2='CZ', reverse=False, gate_round=None):
    regs = range(0, circ.N - 1)
    if reverse:
        regs = reversed(regs)

    for i in regs:
        circ.apply_gate( gate2, i, i + 1, gate_round=gate_round)



def tensor_network_ansatz_circuit(n, depth, gate2='CZ', **kwargs):
    circ = qtn.Circuit(n, **kwargs)

    for r in range(depth):
        # single qubit gate layer
        single_qubit_layer(circ, gate_round=r)

        # alternate between forward and backward CZ layers
        two_qubit_layer(circ, gate2=gate2, gate_round=r, reverse=r % 2 == 0)
        
    single_qubit_layer(circ, gate_round=r + 1)
    return circ



################################################
def loss(circ_unitary, zero_wfn, target_mps):
    """returns -abs(<target_mps|circ_unitary|zero_wfn>)
        assumes that target_mps and zero_wfn are normalized
    """
    return -abs((circ_unitary.H & target_mps & zero_wfn).contract(all, 
                                                                  optimize='auto-hq', 
                                                                  backend='tensorflow'))

    (circ_unitary.H & target_mps).contract(all, optimize='auto-hq', backend='tensorflow')

def quantum_circuit_tensor_network_ansatz(target_mps, depth): 
    """build parametrized circuit from sequential unitary ansatz
    """    
    indx_map = {}
    for it in range(target_mps.L):
        indx_map[f'k{it}']=f'b{it}'
    target_mps = target_mps.reindex(indx_map)
    
    
    quatnum_circ_tn = tensor_network_ansatz_circuit(target_mps.L, depth, gate2='CZ')
    circ_unitary = quatnum_circ_tn.uni
    
    print(f'number of gates in the circuit (from QCTN) are : {quatnum_circ_tn.num_gates}') 
 
    
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

                            