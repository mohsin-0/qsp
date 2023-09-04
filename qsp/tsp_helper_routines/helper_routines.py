# -*- coding: utf-8 -*-
import numpy as np

import quimb as qu
import quimb.tensor as qtn


def compute_energy_expval(psi, qubit_hamiltonian):

    if isinstance(qubit_hamiltonian, qtn.MatrixProductOperator):
        return qtn.expec_TN_1D(psi.H, qubit_hamiltonian, psi) / (psi.H @ psi)

    denom = (psi & psi.H) ^ all
    nrgy = np.zeros(1, np.complex128)
    for indx_top, key in enumerate(qubit_hamiltonian):
        psi_op = psi.copy(deep=True)
        for indx, label in key:
            psi_op.gate_(qu.pauli(label), indx)
        nomin = (psi_op & psi.H) ^ all

        nrgy += qubit_hamiltonian[key] * (nomin / denom)

        # if indx_top%(len(qubit_hamiltonian)//4)==0:
        # print(f'{indx_top}_{nrgy}')

    return np.real(nrgy[0])


def cl_zero_mps(L):
    A = np.zeros((1, 2, 1), dtype=np.complex128)
    A[0, 0, 0] = 1.0
    As = [A for it in range(L)]
    As[0] = A.reshape((1, 2))
    As[-1] = A.reshape((2, 1))
    zero_wfn = qtn.MatrixProductState(As, shape="lpr")
    zero_wfn.permute_arrays(shape="lpr")
    zero_wfn.right_canonize(normalize=True)

    return zero_wfn


def unitaries_specs(Gs_lst):
    gate_count = 0
    depth = 0
    for _, _, Gs, _, _ in Gs_lst:
        curr_depth = 0
        for G in Gs:
            gate_count = gate_count + 1
            curr_depth = curr_depth + 1
        if depth < curr_depth:
            depth = curr_depth

    return depth, gate_count


def unitaries_sanity_check(Gs_list):
    chks = []
    for _, _, Gs, _, _ in Gs_list:
        for G in Gs:
            chks.append(np.allclose(np.eye(G.shape[0]) - G.data @ G.data.T.conj(), 0))
            chks.append(np.allclose(np.eye(G.shape[0]) - G.data.T.conj() @ G.data, 0))
    assert all(chks) == True, "every G in the list should be an unitary"


def norm_mps_ovrlap(mps1, mps2):
    # nomin = mps1.H@mps2
    # denom1= mps1.H@mps1
    # denom2= mps2.H@mps2
    nomin = qtn.TensorNetwork([mps1, mps2.H]) ^ all
    denom1 = qtn.TensorNetwork([mps1, mps1.H]) ^ all
    denom2 = qtn.TensorNetwork([mps2, mps2.H]) ^ all

    return nomin / np.sqrt(denom1 * denom2)


def blockup_mps(mps, block_size):
    assert mps.L / block_size == mps.L // block_size, (
        "mps length block size is not " "exactly divisible by block size"
    )

    tens = []
    for block_indx, row in enumerate(np.reshape(np.arange(mps.L), (-1, block_size))):

        # TODO implement using ncon
        tn = qtn.TensorNetwork()
        for i in row:
            tn.add(mps.tensor_map[i])
        ten = ((tn ^ all).fuse({f"k{block_indx}": [f"k{i}" for i in row]})).data

        if block_indx == 0:
            ten = ten.transpose(1, 0)
        elif len(ten.shape) == 3:
            ten = ten.transpose(0, 2, 1)
        tens.append(ten.data)

    blocked_mps = qtn.MatrixProductState(tens)
    blocked_mps.permute_arrays(shape="lrp")
    return blocked_mps


def make_splitted_mps(tens):
    splitted_tens = []
    for indx, ten in enumerate(tens):
        if indx == 0:
            ten = qtn.Tensor(ten.reshape((2, 2, 2)), inds=("vr", "pl", "pr"))
            tn = ten.split(("pl"), bond_ind="v0")
            ten0 = (tn.tensor_map[0]).transpose(*("v0", "pl"))
            ten1 = (tn.tensor_map[1]).transpose(*("v0", "vr", "pr"))

        elif indx == (len(tens) - 1):
            ten = qtn.Tensor(ten.reshape((2, 2, 2)), inds=("vl", "pl", "pr"))
            tn = ten.split(("vl", "pl"), bond_ind="v0")
            ten0 = (tn.tensor_map[0]).transpose(*("vl", "v0", "pl"))
            ten1 = (tn.tensor_map[1]).transpose(*("v0", "pr"))

        else:
            ten = qtn.Tensor(ten.reshape((2, 2, 2, 2)), inds=("vl", "vr", "pl", "pr"))
            tn = ten.split(("vl", "pl"), bond_ind="v0")
            ten0 = (tn.tensor_map[0]).transpose(*("vl", "v0", "pl"))
            ten1 = (tn.tensor_map[1]).transpose(*("v0", "vr", "pr"))

        splitted_tens.append(ten0.data)
        splitted_tens.append(ten1.data)
    return splitted_tens
