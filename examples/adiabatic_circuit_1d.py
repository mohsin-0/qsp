#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from qsd import qs_decomposition

class Cost():
    def __init__(self, n_gates, n_cx):
        self.n_gates = n_gates 
        self.n_cx = n_cx
    
    def __str__(self):
        return f'n_gates={self.n_gates}, n_cx = {self.n_cx}\n'
        
        
    def __add__(self, other):
        return Cost(self.n_gates + other.n_gates, 
                    self.n_cx + other.n_cx)


def gate_cost(op):
    size = int(np.sqrt(np.prod(op.shape)))
    circ = qs_decomposition(op.reshape(size,size))
    n_gates =  circ.size()
    n_cx = circ.num_nonlocal_gates()
    return Cost(n_gates, n_cx)


def adiabatic_circuit(gates):
    # TODO - implementation is missing
    
    cost = Cost(0,0)
    for t in gates.keys():
        gates_t = gates[t]
        for qubits, gate in gates_t:
            cost = cost + gate_cost(gate)
            print(cost)
    
    return cost
    
if __name__ == "__main__":
    pass