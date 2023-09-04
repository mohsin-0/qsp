#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import numpy as np
import scipy as sp

from ncon import ncon
import qiskit
from qiskit.providers.aer import QasmSimulator


backend = QasmSimulator()
unitary_backend = qiskit.Aer.get_backend("unitary_simulator")


def lcu_circuit_from_unitary_layers(circ, kappas, unitary_layers, target_mps=[]):
    k = int(np.ceil(np.log2(len(kappas))))

    as_circs = copy.deepcopy(unitary_layers)
    kappas = copy.deepcopy(kappas)

    for u_it, unitary_layer in enumerate(unitary_layers):
        for mps_num, (start_indx, end_indx, Gs, _, _) in enumerate(unitary_layer[0]):
            for it in range(start_indx, end_indx + 1):
                qc, phase = qiskit_decomposition(Gs[it - start_indx].data)
                kappas[u_it] = kappas[u_it] * (np.conj(phase))

                as_circs[u_it][0][mps_num][2][it - start_indx] = qc

    prepapre_state(circ, kappas, inverse=False)
    for u_it in range(2**k):
        mcx_gate = qiskit.circuit.library.MCXGate(k, ctrl_state=u_it)

        circ.append(mcx_gate, list(range(k + 1)))
        apply_ctrl_unitary_layer(circ, as_circs[u_it], k)
        circ.append(mcx_gate, list(range(k + 1)))

    prepapre_state(circ, kappas, inverse=True)

    circ_copy = circ.copy()
    overlap_from_lcu_circ = None

    # some sanity check
    if target_mps:
        if target_mps.L <= 16:
            vec = get_state_vector(circ)
            L = target_mps.L
            zero = np.zeros(2 ** (k + 1))
            zero[0] = 1.0
            lcu_output = ncon(
                [vec.reshape(2**L, -1), zero], [(-1, 1), (1,)]
            ).flatten()
            targe_vec = (
                (target_mps ^ all)
                .transpose(*reversed([f"k{i}" for i in range(L)]))
                .data.flatten()
            )
            overlap_from_lcu_circ = overlap(targe_vec, lcu_output)
            # print('overlap to target mps from constructed lcu circuit =',
            #        overlap_from_lcu_circ)

    return circ_copy, overlap_from_lcu_circ


def apply_ctrl_unitary_layer(circ, unitary_layer, k):

    for start_indx, end_indx, Gs, _, _ in unitary_layer[0]:
        for it in range(start_indx, end_indx + 1):
            u = Gs[it - start_indx].data
            if it == end_indx:
                apply_cu(circ, u, k, it)

            else:
                apply_cu(circ, u, k, it + 1, it)


def prepapre_state(circ, kappas, inverse=False):
    temp = np.sqrt(kappas)
    if inverse:
        temp = np.conj(temp)

    temp = temp / np.sqrt(np.conj(temp.T) @ temp)
    state_prep = qiskit.circuit.library.StatePreparation(temp, inverse=inverse)
    circ.append(state_prep, list(range(state_prep.num_qubits)))


def qiskit_decomposition(u):
    num_qubits = int(np.log2(len(u)))
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.unitary(u, range(num_qubits))
    qc = qiskit.transpile(qc, basis_gates=["cx", "u3"], optimization_level=0)

    # print(np.sum(np.abs(get_unitary(qc) - u)))
    # if u.shape==(2,2):
    # return qc, 1

    # determine the phase missing from qiskit decomposition/transpilation
    # u = phase* (circ made of basis gates i.e. u_prime)
    def u3(th, phi, lam):
        th = th / 2
        u3 = np.array(
            [
                [np.cos(th), -np.exp(1j * lam) * np.sin(th)],
                [np.exp(1j * phi) * np.sin(th), np.exp(1j * (phi + lam)) * np.cos(th)],
            ]
        )
        return u3

    cz = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ]
    )

    cx = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    num_qubits = qc.num_qubits

    u_prime = np.eye(2**num_qubits)
    u_prime = u_prime.reshape([2] * (num_qubits * 2))

    for instruction in qc:
        if instruction.operation.name == "u3":
            qubit_indx = instruction.qubits[0].index
            params = instruction.operation.params
            th, phi, lam = params

            if isinstance(th, qiskit.circuit.parameterexpression.ParameterExpression):
                th = th

            qubit_indx = [1, 0][instruction.qubits[0].index]
            inds = [-i for i in range(1, 2 * num_qubits + 1)]

            inds[qubit_indx] = -inds[qubit_indx]
            u_prime = ncon(
                [u_prime, (u3(th, phi, lam))],
                (list(inds), [-(qubit_indx + 1), qubit_indx + 1]),
            )

        elif instruction.operation.name == "cx":
            # u_prime = cz@u_prime

            qubit_indx1 = [1, 0][instruction.qubits[0].index]
            qubit_indx2 = [1, 0][instruction.qubits[1].index]

            inds = [-i for i in range(1, 2 * num_qubits + 1)]
            inds[qubit_indx1] = -inds[qubit_indx1]
            inds[qubit_indx2] = -inds[qubit_indx2]

            u_prime = ncon(
                [u_prime, cx.reshape(2, 2, 2, 2)],
                [
                    inds,
                    [
                        qubit_indx1 + 1,
                        qubit_indx2 + 1,
                        -(qubit_indx1 + 1),
                        -(qubit_indx2 + 1),
                    ],
                ],
            )

    u_prime = u_prime.reshape(2**num_qubits, 2**num_qubits)
    phases = []
    for r1, r2 in zip(u_prime, u):
        for a, b in zip(r1, r2):
            if (np.abs(a) > 1e-8) and (np.abs(b) > 1e-8):
                phases.append(a / b)

    # assert np.allclose(np.array(phases)-phases[0],
    #                    np.zeros(len(phases))), ('problem with phase extraction '
    #                                             'all phases are not same')

    return qc, phases[0]


def apply_cu(circ, unitary_as_circuit, k, orig_indx, orig_indx_plus1=[]):

    for instruction in unitary_as_circuit:
        if instruction.operation.name == "u3":
            qubit_indx = [orig_indx, orig_indx_plus1][instruction.qubits[0].index]
            params = instruction.operation.params
            gate = qiskit.circuit.library.CU3Gate(*params)

            circ.append(gate, [k, (k + 1) + qubit_indx])

        elif instruction.operation.name == "cx":
            qubit1 = [orig_indx, orig_indx_plus1][instruction.qubits[0].index]
            qubit2 = [orig_indx, orig_indx_plus1][instruction.qubits[1].index]
            circ.ccx(k, (k + 1) + qubit1, (k + 1) + qubit2)

        else:
            exit()


def apply_lcu(circ, kappas, unitaries_as_circuits, target_mps=[]):
    k = int(np.ceil(np.log2(len(kappas))))

    prepapre_state(circ, kappas, inverse=False)
    for u_it in range(2**k):
        mcx_gate = qiskit.circuit.library.MCXGate(k, ctrl_state=u_it)

        circ.append(mcx_gate, list(range(k + 1)))
        apply_cu(circ, unitaries_as_circuits[u_it], k, 0, 1)
        circ.append(mcx_gate, list(range(k + 1)))

    prepapre_state(circ, kappas, inverse=True)


###############################################################################


def get_unitary(circ):
    job = qiskit.execute(circ, unitary_backend, shots=1)
    result = job.result()
    return result.get_unitary(circ)


def get_state_vector(circ):
    circ.save_statevector(label="vec_final")

    backend_options = {"method": "statevector"}
    job = qiskit.execute(circ, backend, backend_options=backend_options)
    result = job.result()
    vec = result.data()["vec_final"].data

    return vec.reshape([2] * circ.num_qubits)


def normalize(vec):
    return vec / np.abs(np.sqrt(np.conj(vec).T @ vec))


def overlap(vec1, vec2):
    vec1 = normalize(vec1)
    vec2 = normalize(vec2)
    return np.abs(np.conj(vec1).T @ vec2)


###############################################################################

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # pass
    k = 2  # number of ancilla qubits
    L = 2  # number of target qubits
    num_qubits = L + k + 1

    unitaries = [
        (
            sp.linalg.svd(
                np.random.rand(2**L, 2**L) + np.random.rand(2**L, 2**L) * 1j
            )
        )[0]
        for _ in range(2**k)
    ]

    kappas = np.random.rand(4) + 1j * np.random.rand(4) * 1  #
    kappas = np.array(kappas, np.complex128)

    uni = unitaries.copy()
    kp = kappas.copy()

    unitaries_as_circuits = {}
    for u_it in range(2**k):
        qc, phase = qiskit_decomposition(unitaries[u_it])
        unitaries_as_circuits[u_it] = qc
        kappas[u_it] = kappas[u_it] * (np.conj(phase))

    circ = qiskit.QuantumCircuit(num_qubits)
    apply_lcu(circ, kappas, unitaries_as_circuits)

    vec = get_state_vector(circ)
    zero = np.zeros(2 ** (k + 1))
    zero[0] = 1
    lcu_output = ncon([vec.reshape(2**L, -1), zero], [(-1, 1), (1,)]).flatten()

    unitary_sum = np.sum(
        np.array([kappa * unitary for kappa, unitary in zip(kp, uni)]), axis=0
    )
    zero = np.zeros(2**L)
    zero[0] = 1.0
    print("final overlap: ", np.abs(overlap(unitary_sum @ zero, lcu_output)))
