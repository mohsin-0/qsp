#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

from ncon import ncon

from qiskit.providers.aer import QasmSimulator
import qiskit

backend = QasmSimulator()
unitary_backend = qiskit.Aer.get_backend('unitary_simulator')


def prepapre_state(circ, kappas, inverse=False):
    temp = np.sqrt(kappas)
    
    temp = temp/np.sqrt(np.conj(temp.T)@temp)
    state_prep = qiskit.circuit.library.StatePreparation(temp, inverse=inverse)    
    circ.append(state_prep, list(range(state_prep.num_qubits)) )
    

def qiskit_decomposition(u):
    num_qubits = int(np.log2(len(u)))
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.unitary(u, range(num_qubits))
    qc = qiskit.transpile(qc, basis_gates=['cx','u3'])
    return qc
    

def apply_cu(circ, u, k, orig_indx, orig_indx_plus1=[]):
    
    for instruction in qiskit_decomposition(u):
        if instruction.operation.name=='u3':
            qubit_indx = [orig_indx, orig_indx_plus1][instruction.qubits[0].index]
            params = instruction.operation.params
            gate = qiskit.circuit.library.CU3Gate(*params)
            
            circ.append(gate, [k, (k+1)+qubit_indx])
            
        elif instruction.operation.name=='cx':
            qubit1 = [orig_indx, orig_indx_plus1][instruction.qubits[0].index]
            qubit2 = [orig_indx, orig_indx_plus1][instruction.qubits[1].index]
            circ.ccx(k, (k+1)+qubit1, (k+1)+qubit2)
            
        else:
            exit()

            
def apply_ctrl_unitary_layer(circ, unitary_layer, k):
    
    for start_indx, end_indx, Gs, _, _ in unitary_layer[0]:
        for it in range(start_indx, end_indx+1):
            u = Gs[it-start_indx].data
            if (it==end_indx):
                apply_cu(circ, u, k, it)

            else:                
                apply_cu(circ, u, k, it+1, it)    



def apply_lcu_with_layers(circ, kappas,  unitaries, target_mps=[]):
    k = int(np.ceil(np.log2(len(kappas))))
    
    prepapre_state(circ, kappas, inverse=False)
    for u_it in range(2**k):        
        mcx_gate = qiskit.circuit.library.MCXGate(k, ctrl_state=u_it)
        
        circ.append(mcx_gate, list(range(k+1)))
        apply_ctrl_unitary_layer(circ, unitaries[u_it], k )
        circ.append(mcx_gate, list(range(k+1)))
        
    prepapre_state(circ, kappas, inverse=True)
    
    circ_copy = circ.copy()
    
    #3 some sanity check
    if target_mps:
        vec = get_state_vector(circ)    
        L = target_mps.L
        zero = np.zeros(2**(k+1))
        zero[0] = 1.
        lcu_output = ncon([vec.reshape(2**L, -1), zero], [(-1,1),(1,)]).flatten()

        targe_vec = (target_mps^all).transpose(*reversed([f'k{i}' for i in range(L)])).data.flatten()
    
        print('overlap to target mps from constructed lcu circuit =', 
              overlap(targe_vec, lcu_output))    
    
    return circ_copy

def apply_lcu(circ, kappas,  unitaries, target_mps=[]):
    k = int(np.ceil(np.log2(len(kappas))))
    
    prepapre_state(circ, kappas, inverse=False)
    for u_it in range(2**k):  
        mcx_gate = qiskit.circuit.library.MCXGate(k, ctrl_state=u_it)
        
        circ.append(mcx_gate, list(range(k+1)))
        apply_cu(circ, unitaries[u_it], k, 0,1)
        circ.append(mcx_gate, list(range(k+1)))
        
    prepapre_state(circ, kappas, inverse=True)
    
    
    
###############################################################################
    
def get_unitary(circ):
    job = qiskit.execute(circ, unitary_backend, shots=1)
    result = job.result()
    return result.get_unitary(circ)


def get_state_vector(circ):
    circ.save_statevector(label='vec_final')
    
    backend_options = {'method': 'statevector'}
    job = qiskit.execute(circ, backend, backend_options=backend_options)
    result = job.result()
    vec = result.data()['vec_final'].data

    return vec.reshape([2]*circ.num_qubits)


def normalize(vec):
    return vec/np.abs(np.sqrt(np.conj(vec).T@vec))


def overlap(vec1, vec2):
    vec1 = normalize(vec1)
    vec2 = normalize(vec2)
    return np.abs(np.conj(vec1).T@vec2)

###############################################################################



if __name__ == "__main__":    
    k = 2 # number of ancilla qubits
    L = 2 # number of target qubits
    num_qubits = L+k+1
    
    u = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    
    L=L-1
    unitaries = [(sp.linalg.svd(np.random.rand(2**L, 2**L)+ 
                                np.random.rand(2**L, 2**L)*1j))[0] for _ in range(2**k)]        
    unitaries = [np.kron(u,u) for u in unitaries]
    L = L+1
    unitaries = [u*np.exp(-1j*np.angle(np.diag(u))) for u in unitaries]
    kappas = np.random.rand(4)#[1,0,0,0]#
    kappas = np.array(kappas, np.complex128)

        
    circ = qiskit.QuantumCircuit(num_qubits)
    apply_lcu(circ, kappas,  unitaries)

    vec = get_state_vector(circ)    
    zero = np.zeros(2**(k+1))
    zero[0] = 1
    lcu_output = ncon([vec.reshape(2**L, -1), zero], [(-1,1),(1,)]).flatten()

    unitary_sum = np.sum(np.array([kappa*unitary for kappa, unitary in zip(kappas, unitaries)]), axis=0)
    zero = np.zeros(2**L)
    zero[0] = 1.
    print('overlap: ', np.abs(overlap(unitary_sum@zero, lcu_output)))
    
    