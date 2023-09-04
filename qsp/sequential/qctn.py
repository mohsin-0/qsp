#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import quimb as qu
import quimb.tensor as qtn

from ..q_circs import circuit_from_quimb_unitary

from ..tsp_helper_routines import cl_zero_mps
from ..tsp_helper_routines import norm_mps_ovrlap


def overlap_value(al, psi, approx_psi, residual, qubit_hamiltonian=0):
    return -np.abs(norm_mps_ovrlap(psi, approx_psi + al[0] * residual))


# quantum circuit tensor network ansatz
def single_qubit_layer(circ, gate_round=None, gid_to_qubit=None):
    """Apply a parametrizable layer of single qubit ``U3`` gates."""
    count = len(gid_to_qubit.keys())
    for i in range(circ.N):
        # initialize with random parameters
        params = qu.randn(3, dist="uniform")
        circ.apply_gate("U3", *params, i, gate_round=gate_round, parametrize=True)

        gid_to_qubit[count] = (i,)
        count = count + 1


def two_qubit_layer(circ, gate2, reverse=False, gate_round=None, gid_to_qubit=None):

    count = len(gid_to_qubit.keys())
    regs = range(0, circ.N - 1)
    if reverse:
        regs = reversed(regs)

    for i in regs:
        circ.apply_gate(gate2, i, i + 1, gate_round=gate_round)
        # if reverse:
        gid_to_qubit[count] = (i, i + 1)
        # else:
        # gid_to_qubit[count] = (i, i+1)

        count = count + 1


def tensor_network_ansatz_circuit(n, depth, gate2, **kwargs):
    circ = qtn.Circuit(n, **kwargs)

    gid_to_qubit = {}

    for r in range(depth):
        # single qubit gate layer
        single_qubit_layer(circ, gate_round=r, gid_to_qubit=gid_to_qubit)

        # alternate between forward and backward CZ layers
        two_qubit_layer(
            circ,
            gate2=gate2,
            gate_round=r,
            reverse=(r % 2 == 0),
            gid_to_qubit=gid_to_qubit,
        )

    single_qubit_layer(circ, gate_round=r + 1, gid_to_qubit=gid_to_qubit)
    return circ, gid_to_qubit


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


def quantum_circuit_tensor_network_ansatz(
    target_mps, depth, n_iter, nhop, verbose=False
):
    """build parametrized circuit from sequential unitary ansatz"""
    indx_map = {}
    for it in range(target_mps.L):
        indx_map[f"k{it}"] = f"b{it}"
    target_mps = target_mps.reindex(indx_map)

    quatnum_circ_tn, gid_to_qubit = tensor_network_ansatz_circuit(
        target_mps.L, depth, gate2="CX"
    )
    circ_unitary = quatnum_circ_tn.get_uni(transposed=True)  # quatnum_circ_tn.uni

    zero_wfn = cl_zero_mps(target_mps.L)
    zero_wfn = zero_wfn.astype(np.complex64)
    target_mps = target_mps.astype(np.complex64)
    circ_unitary = circ_unitary.astype(np.complex64)

    print(
        f"number of variational params in the circuit (from QCTN) are "
        f"{quatnum_circ_tn.num_gates*3}"
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
        tags=["U3"],  # only optimize RX and RZ tensors
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
