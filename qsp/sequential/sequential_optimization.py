#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


import qiskit
import quimb.tensor as qtn

from ..q_circs import circuit_from_quimb_unitary

from ..tsp_helper_routines import cl_zero_mps
from ..tsp_helper_routines import norm_mps_ovrlap


def overlap_value(al, psi, approx_psi, residual, qubit_hamiltonian=0):
    return -np.abs(norm_mps_ovrlap(psi, approx_psi + al[0] * residual))


def apply_quimb_unitary(u, quimb_circ, it, it_plus_1, gid_to_qubit):

    params_count = 0
    n_qubits = int(np.log2(u.shape[0]))

    circ = qiskit.QuantumCircuit(n_qubits)
    circ.unitary(u, list(range(n_qubits)))

    if n_qubits == 1:
        # single qubit unitary, to take care of the reverse ordering in the single-case
        it_plus_1 = it

    trans_qc = qiskit.transpile(circ, basis_gates=["cx", "u3"])

    count = len(gid_to_qubit.keys())

    for instruction in trans_qc:
        if instruction.operation.name == "u3":
            qubit_id = (it_plus_1, it)[instruction.qubits[0].index]
            params = instruction.operation.params
            quimb_circ.apply_gate(
                "U3", params[0], params[1], params[2], qubit_id, parametrize=True
            )

            gid_to_qubit[count] = (qubit_id,)
            params_count = params_count + 3

        elif instruction.operation.name == "cx":
            quimb_circ.apply_gate("CX", it_plus_1, it)
            gid_to_qubit[count] = (it_plus_1, it)

        else:
            print("unhandled gate")
            exit()
        count = count + 1

    return params_count


def generate_circ_from_unitary_layers(unitary_layers, L):
    gid_to_qubit = {}
    ttl_params_count = 0
    quimb_circ = qtn.Circuit(L)
    for it in reversed(range(len(unitary_layers))):
        Gs_lst = unitary_layers[it]
        for start_indx, end_indx, Gs, _, _ in Gs_lst:
            for it in range(start_indx, end_indx + 1):
                if it == end_indx:
                    params_count = apply_quimb_unitary(
                        Gs[it - start_indx].data, quimb_circ, it, it + 1, gid_to_qubit
                    )

                else:
                    params_count = apply_quimb_unitary(
                        Gs[it - start_indx].data, quimb_circ, it, it + 1, gid_to_qubit
                    )
                ttl_params_count += params_count

    return quimb_circ, gid_to_qubit, ttl_params_count


################################################
def loss(circ_unitary, zero_wfn, target_mps):
    """returns -abs(<target_mps|circ_unitary|zero_wfn>)
    assumes that target_mps and zero_wfn are normalized
    """
    return -abs(
        (circ_unitary.H & target_mps & zero_wfn).contract(
            all, optimize="auto-hq", backend="tensorflow"
        )
    )


def sequential_unitary_circuit_optimization(
    target_mps, unitaries, n_iter, nhop, verbose=False
):
    """build parametrized circuit from sequential unitary ansatz"""
    quimb_circ, gid_to_qubit, ttl_params_count = generate_circ_from_unitary_layers(
        unitaries, target_mps.L
    )

    indx_map = {}
    for it in range(target_mps.L):
        indx_map[f"k{it}"] = f"b{it}"
    target_mps = target_mps.reindex(indx_map)

    # circ_unitary = quimb_circ.uni
    circ_unitary = quimb_circ.get_uni(transposed=True)

    zero_wfn = cl_zero_mps(target_mps.L)
    zero_wfn = zero_wfn.astype(np.complex64)
    target_mps = target_mps.astype(np.complex64)
    circ_unitary = circ_unitary.astype(np.complex64)

    # if verbose:
    print(
        f"number of variational params in the circuit (from sequential algorithm) are "
        f"{ttl_params_count}"
    )
    print(
        "overlap before variational optimization = "
        f"{loss(circ_unitary, zero_wfn, target_mps):.10f}"
    )

    tnopt = qtn.TNOptimizer(
        circ_unitary,  # tensor network to optimize
        loss,  # function to minimize
        loss_constants={
            "zero_wfn": zero_wfn,  # static inputs
            "target_mps": target_mps,
        },
        tags=["U3"],
        autodiff_backend="tensorflow",
        optimizer="L-BFGS-B",
    )
    optimized_unitary = tnopt.optimize_basinhopping(n=n_iter, nhop=nhop)

    circ, overlap_from_seq_circ = circuit_from_quimb_unitary(
        optimized_unitary, gid_to_qubit, target_mps.L, target_mps
    )
    data = {
        "tnopt": tnopt,
        "optimized_unitary": optimized_unitary,
        "circ": circ,
        "overlap_from_seq_circ": overlap_from_seq_circ,
    }

    return data


if __name__ == "__main__":
    pass
    # mps_type = 'random'#'molecule'#
    # L = 16
    # italic_D_sequ = 4
    # with open(f'data_preparations/sequ_{mps_type}_L{L}_layers{italic_D_sequ}.pkl', 'rb') as f:
    #     [target_mps, unitaries, quimb_overlap] = pkl.load(f)
    # print('quimb_overlap: ', quimb_overlap)
    # unitary_circuit_optimization(target_mps, unitaries)
