#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import random

import qiskit
from qiskit.providers.aer import QasmSimulator
import quimb.tensor as qtn


def approximate_adiabatic_cost(gates):
    def qiskit_decomposition(u):
        num_qubits = int(np.log2(len(u)))
        qc = qiskit.QuantumCircuit(num_qubits)
        qc.unitary(u, range(num_qubits))
        qc = qiskit.transpile(qc, basis_gates=["cx", "u"])
        return qc

    counts = []
    for sampled_gate in random.sample(gates, 30):
        sampled_gate = sampled_gate.reshape(np.prod(sampled_gate.shape[:2]), -1)
        qc = qiskit_decomposition(sampled_gate)
        counts.append([qc.size(), qc.num_nonlocal_gates()])
    return len(gates) * np.mean(counts, axis=0)


def closest_unitary(A):
    V, _, Wh = sp.linalg.svd(A)
    U = V.dot(Wh)
    return U


def circuit_from_unitary_layers(unitary_layers, L, target_mps=[]):
    circ = qiskit.QuantumCircuit(L)

    def apply_unitary_layer(Gs_lst):
        for start_indx, end_indx, Gs, _, _ in Gs_lst:
            for it in range(start_indx, end_indx + 1):
                u = Gs[it - start_indx].data
                if it == end_indx:
                    circ.unitary(u, [it])

                else:
                    circ.unitary(u, [it + 1, it])

    for it in reversed(range(len(unitary_layers))):
        apply_unitary_layer(unitary_layers[it])

    circ = qiskit.transpile(circ, basis_gates=["cx", "u3"])

    overlap_from_seq_circ = None
    if L <= 16 and isinstance(
        target_mps, qtn.tensor_1d.MatrixProductState
    ):  # some snaity check
        circ.save_statevector(label="vec_final")
        backend = QasmSimulator()
        backend_options = {"method": "statevector"}
        job = qiskit.execute(circ, backend, backend_options=backend_options)
        result = job.result()
        qiskit_vec = result.data()["vec_final"].data

        orig = (
            (target_mps ^ all).fuse({"k": (f"k{i}" for i in range(L))}).data
        ).reshape([2] * L)
        orig = orig.transpose(list(reversed(range(L)))).flatten()
        overlap_from_seq_circ = np.abs(np.conj(orig.T) @ qiskit_vec)
        # print(f'overlap to target mps from constructed circuit = {np.abs(np.conj(orig.T)@qiskit_vec)}, ')

    return circ, overlap_from_seq_circ


def find_tensor(u_gen, tag):
    for gate in u_gen:
        if tag in gate.tags:
            return gate

    return "gate not found"


def circuit_from_quimb_unitary(unitary, gid_to_qubit, L, target_mps=[]):
    circ = qiskit.QuantumCircuit(L)

    for gid, qubits in gid_to_qubit.items():
        ten = find_tensor(unitary, "GATE_" + str(gid))

        if len(qubits) == 2:
            if "CZ" in ten.tags:
                circ.cz(qubits[0], qubits[1])
            else:
                circ.cx(qubits[0], qubits[1])

        else:
            circ.unitary(
                closest_unitary(np.array(ten.data, dtype=np.complex128)), [qubits[0]]
            )

    circ = qiskit.transpile(circ, basis_gates=["cx", "u3"])

    overlap_from_seq_circ = None
    ###
    if L <= 16 and isinstance(
        target_mps, qtn.tensor_1d.MatrixProductState
    ):  # some snaity check
        circ.save_statevector(label="vec_final")
        backend = QasmSimulator()
        backend_options = {"method": "statevector"}
        job = qiskit.execute(circ, backend, backend_options=backend_options)
        result = job.result()
        qiskit_vec = result.data()["vec_final"].data

        orig = (
            (target_mps ^ all).fuse({"b": (f"b{i}" for i in range(L))}).data
        ).reshape([2] * L)
        orig = orig.transpose(list(reversed(range(L)))).flatten()
        overlap_from_seq_circ = np.abs(np.conj(orig.T) @ qiskit_vec)
        # print(f'overlap to target mps from constructed circuit = {np.abs(np.conj(orig.T)@qiskit_vec)}, ')

    return circ, overlap_from_seq_circ
