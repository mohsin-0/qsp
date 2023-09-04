#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

from ncon import ncon
import quimb.tensor as qtn
import quimb as qu
from tqdm import tqdm


def make_evolution_mpo(hamiltonian, L, phy_dim, tau, Tmax, compress=True):

    gates = []
    d = phy_dim

    # mpo for applying gates from left to right
    mpo_tens = [[] for _ in range(L)]
    for bond in hamiltonian.keys():
        evo_op = sp.linalg.expm(
            -hamiltonian[bond].reshape(d**2, d**2) * 1j * tau * Tmax / 2
        ).reshape((d, d, d, d))
        u, s, v = sp.linalg.svd(evo_op.transpose([0, 2, 1, 3]).reshape(d**2, d**2))
        v = np.diag(s) @ v

        mpo_tens[bond[0]].append(u.reshape(d, d, -1))
        mpo_tens[bond[1]].append(v.reshape((-1, d, d)))  # first comes v and then u

        gates.append(evo_op)

    for indx in range(L):  # run over all the sites
        if indx == 0:
            mpo_tens[0] = mpo_tens[0][0].transpose((2, 0, 1))

        elif indx == (L - 1):
            mpo_tens[L - 1] = mpo_tens[L - 1][0]

        else:
            v, u = mpo_tens[indx][0], mpo_tens[indx][1]
            mpo_tens[indx] = ncon((v, u), ([-1, -3, 1], [1, -4, -2]))

    mpo_1 = qtn.MatrixProductOperator(mpo_tens, "lrdu")

    # mpo for applying gates from right to left
    mpo_tens = [[] for _ in range(L)]
    for bond in reversed(hamiltonian.keys()):  # run over all the interaction
        evo_op = sp.linalg.expm(
            -hamiltonian[bond].reshape(d**2, d**2) * 1j * tau * Tmax / 2
        ).reshape((d, d, d, d))
        u, s, v = sp.linalg.svd(evo_op.transpose([0, 2, 1, 3]).reshape(d**2, d**2))
        v = np.diag(s) @ v

        mpo_tens[bond[0]].append(u.reshape(d, d, -1))  # first comes u and then u
        mpo_tens[bond[1]].append(v.reshape((-1, d, d)))

        gates.append(evo_op)

    for indx in range(L):  # run over all the sites
        if indx == 0:
            mpo_tens[0] = mpo_tens[0][0].transpose((2, 0, 1))

        elif indx == (L - 1):
            mpo_tens[L - 1] = mpo_tens[L - 1][0]

        else:
            u, v = mpo_tens[indx][0], mpo_tens[indx][1]
            mpo_tens[indx] = ncon((u, v), ([-3, 1, -2], [-1, 1, -4]))

    mpo_2 = qtn.MatrixProductOperator(mpo_tens, "lrdu")

    if compress:
        mpo_1.compress(cutoff=1e-12, cutoff_mode="sum2")
        mpo_2.compress(cutoff=1e-12, cutoff_mode="sum2")

    return mpo_1, mpo_2, gates


def make_hamiltonian_mpo(hamiltonian, L, phy_dim, compress=False):
    d = phy_dim

    H = qtn.SpinHam1D(S=(d - 1) / 2)
    for bond in hamiltonian.keys():
        u, s, v = sp.linalg.svd(
            hamiltonian[bond].transpose([0, 2, 1, 3]).reshape(d**2, d**2)
        )

        for it in range(s.shape[0]):
            if np.abs(s[it]) > 1e-12:
                H[bond[0], bond[1]] += (
                    s[it],
                    qu.qu(u[:, it].reshape(d, d)),
                    qu.qu(v[it, :].reshape(d, d)),
                )

    H_local_ham1D = H.build_local_ham(L)
    H_mpo = H.build_mpo(
        L,
    )
    if compress is True:
        H_mpo.compress(cutoff=1e-12, cutoff_mode="sum2")

    return H_mpo, H_local_ham1D


def constuct_parent_hamiltonian(Qs, L, phy_dim):
    d = phy_dim

    bonds = [(it, it + 1) for it in range(L - 1)]

    hamiltonian = {}
    for (site1, site2) in bonds:
        Q1, Q2 = Qs[site1], Qs[site2]

        if Q1.ndim == 2:
            Q1 = np.expand_dims(Q1, axis=0)

        if Q2.ndim == 2:
            Q2 = np.expand_dims(Q2, axis=1)

        kernel = sp.linalg.null_space(
            ncon([Q1, Q2], [(-1, 1, -3), (1, -2, -4)]).reshape(-1, d**2)
        )

        ham_term = 0.0
        for it in range(kernel.shape[1]):
            v = kernel[:, it]
            ham_term += ncon((np.conj(v), v), ([-1], [-2]))
        hamiltonian[(site1, site2)] = ham_term.reshape((d, d, d, d))

    return hamiltonian


def calculate_energy_from_parent_hamiltonian_mpo(mps, H_mpo):
    mps_adj = mps.H
    mps_adj.reindex({f"k{i}": f"b{i}" for i in range(mps.L)}, inplace=True)
    # mps_adj.align_(H_mpo, mps_adj)
    exp_val = ((mps_adj & H_mpo & mps) ^ all) / ((mps.H & mps) ^ all)
    return exp_val


def adiabatic_state_preparation_1d(
    target_mps, initial_mps, Tmax, tau, s_func, max_bond, verbose=False
):
    L = target_mps.L
    phy_dim = target_mps.phys_dim()

    target_mps.permute_arrays(shape="lrp")
    target_mps.normalize()  # inplace

    initial_mps.permute_arrays(shape="lrp")
    initial_mps.normalize()

    ts = np.arange(0, Tmax + tau, tau)
    ss, energy, target_fidelity, current_fidelity = {}, {}, {}, {}
    gates = {}

    psi = initial_mps
    for t in tqdm(ts):
        s = s_func(t)

        Qs = [0] * L
        for site, (initial, target) in enumerate(zip(initial_mps, target_mps)):
            Qs[site] = (1 - s) * initial.data + s * target.data

        hamiltonian = constuct_parent_hamiltonian(Qs, L, phy_dim)
        hamiltonian_mpo, _ = make_hamiltonian_mpo(
            hamiltonian, L, phy_dim, compress=False
        )
        mpo_lr, mpo_rl, gates_curr = make_evolution_mpo(
            hamiltonian, L, phy_dim, tau, Tmax, compress=True
        )

        mps_s = qtn.MatrixProductState(Qs, shape="lrp")
        mps_s.normalize()

        # sanity check to see if the mps is indeed the ground state of the parent hamiltonian
        assert (
            calculate_energy_from_parent_hamiltonian_mpo(mps_s, hamiltonian_mpo) / L
            < 1e-14
        )

        ## apply second order trotter decomposition
        psi = mpo_lr.apply(psi)
        psi.compress(max_bond=max_bond)

        psi = mpo_rl.apply(psi)
        psi.compress(max_bond=max_bond)

        psi.normalize()

        ###
        ss[t] = s
        energy[
            t
        ] = 0  # np.real(calculate_energy_from_parent_hamiltonian_mpo(psi, hamiltonian_mpo))
        target_fidelity[t] = np.abs(((target_mps.H & psi) ^ all))
        current_fidelity[t] = np.abs(((mps_s.H & psi) ^ all))
        gates[t] = gates_curr

        if verbose:
            print(
                f"\n{t=:.2f}, {s=:.4f}, e={energy[t]:.4f}, "
                f"f={target_fidelity[t]:.8f}, curr_f={current_fidelity[t]:.8f}\n"
            )

    ###################################
    data = {
        "ss": ss,
        "energy": energy,
        "target_fidelity": target_fidelity,
        "current_fidelity": current_fidelity,
        "gates": gates,
    }
    return data


def main():
    L = 8
    Tmax, tau = 6, 0.04  # total runtime, trotter step size
    max_bond = 2

    s_func = lambda t: np.sin((np.pi / 2) * np.sin((np.pi / 2) * t / Tmax) ** 2) ** 2
    # s_func = lambda t: np.sin( (np.pi/2)*t/Tmax)**2
    # s_func = lambda t: t/Tmax

    # # ####################################
    from misc_states import make_aklt_mps, make_bell_pair_mps

    target_tens, _ = make_aklt_mps(L=L)
    initial_tens = make_bell_pair_mps(L=L, phys_dim=4)

    initial_mps = qtn.MatrixProductState(initial_tens, shape="lrp")
    target_mps = qtn.MatrixProductState(target_tens, shape="lrp")

    data = adiabatic_state_preparation_1d(
        target_mps, initial_mps, Tmax, tau, s_func, max_bond
    )

    t_last = max(data["ss"].keys())
    s, e = data["ss"][t_last], data["energy"][t_last]
    curr_f, tar_f = (
        data["current_fidelity"][t_last],
        data["target_fidelity"][t_last],
    )
    print(
        f"\nfinal overlap is {s=:.5f}, e={e:.08f}, "
        f"curr_f={curr_f:.08f}, target_fid={tar_f:.08f}\n"
    )

    from matplotlib import pyplot as plt

    x, y = data["target_fidelity"].keys(), data["target_fidelity"].values()
    plt.plot(x, y, ".-")

    x, y = data["ss"].keys(), data["ss"].values()
    plt.plot(x, y, ".-")


if __name__ == "__main__":
    main()
