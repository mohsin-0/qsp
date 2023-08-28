#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

from ncon import ncon

from qiskit.providers.aer import QasmSimulator
import qiskit

backend = QasmSimulator()
unitary_backend = qiskit.Aer.get_backend('unitary_simulator')


class Cost():
    def __init__(self, n_gates, n_cx):
        self.n_gates = n_gates 
        self.n_cx = n_cx
    
    def __str__(self):
        return f'n_gates={self.n_gates}, n_cx = {self.n_cx}\n'
        
        
    def __add__(self, other):
        return Cost(self.n_gates + other.n_gates, 
                    self.n_cx + other.n_cx)


# def gate_cost(op):
#     size = int(np.sqrt(np.prod(op.shape)))
#     circ = qs_decomposition(op.reshape(size,size))
#     n_gates =  circ.size()
#     n_cx = circ.num_nonlocal_gates()
#     return Cost(n_gates, n_cx)


def adiabatic_circuit(gates):
    # TODO - implementation is missing
    
    # cost = Cost(0,0)
    # for t in gates.keys():
    #     gates_t = gates[t]
    #     for qubits, gate in gates_t:
    #         cost = cost + gate_cost(gate)
    #         print(cost)
    
    # return cost
    pass


def qiskit_decomposition(u):
    num_qubits = int(np.log2(len(u)))
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.unitary(u, range(num_qubits))
    qc = qiskit.transpile(qc, basis_gates=['cx','u'])
    return qc
    

def apply_control_unitary(circ, u, ctrl_state, k):

    multi_control_x_gate = qiskit.circuit.library.MCXGate(k, ctrl_state=ctrl_state)
    circ.append(multi_control_x_gate, list(range(k+1)))
    
    # print(u-get_unitary(qc1))
    for instruction in qiskit_decomposition(u):
        if instruction.operation.name=='u':
            qubit_indx = instruction.qubits[0].index
            params = instruction.operation.params
            gate = qiskit.circuit.library.CUGate(*params, 0)
            
            circ.append(gate, [k, (k+1)+qubit_indx])
            
        elif instruction.operation.name=='cx':
            qubit1, qubit2 = instruction.qubits[0].index, instruction.qubits[1].index
            circ.ccx(k, (k+1)+qubit1, (k+1)+qubit2)
            
        else:
            exit()
    
    
    # circ.unitary(custom, [k, (k+1)+0, (k+1)+1])
    circ.append(multi_control_x_gate, list(range(k+1)))
    

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


def u_decom(u):
    qc1 = qiskit.QuantumCircuit(1)
    qc1.unitary(u, [0])
    qc1 = qiskit.transpile(qc1, basis_gates=['cx','u3'])
    params = (list(qc1)[0]).operation.params
    
    print(u-get_unitary(qc1))
    return params


def normalize(vec):
    return vec/np.abs(np.sqrt(np.conj(vec).T@vec))


def overlap(vec1, vec2):
    vec1 = normalize(vec1)
    vec2 = normalize(vec2)
    return np.abs(np.conj(vec1).T@vec2)

def prepapre_state(circ, kappas, inverse=False):
    temp = np.sqrt(kappas)
    
    temp = temp/np.sqrt(np.conj(temp.T)@temp)
    state_prep = qiskit.circuit.library.StatePreparation(temp, inverse=inverse)    
    circ.append(state_prep, list(range(state_prep.num_qubits)) )


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


    kappas = np.random.rand(4)

    circ = qiskit.QuantumCircuit(num_qubits)
    prepapre_state(circ, kappas, inverse=False)
    for u_it in range(2**k):        
        apply_control_unitary(circ, unitaries[u_it], ctrl_state=u_it, k=k)
    prepapre_state(circ, kappas, inverse=True)
    
    vec = get_state_vector(circ)    
    zero = np.zeros(2**(k+1))
    zero[0] = 1
    lcu_output = ncon([vec.reshape(2**L, -1), zero], [(-1,1),(1,)]).flatten()

    unitary_sum = np.sum(np.array([kappa*unitary for kappa, unitary in zip(kappas, unitaries)]), axis=0)
    zero = np.zeros(2**L)
    zero[0] = 1.
    print('overlap: ', np.abs(overlap(unitary_sum@zero, lcu_output)))
    