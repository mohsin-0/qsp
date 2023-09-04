#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

from ncon import ncon
import quimb as qu
import quimb.tensor as qtn
from tqdm import tqdm


def construct_parent_hamiltonian(tensor_grid, bonds, Lx, Ly, phy_dim):
    d = phy_dim

    num2pt = {Lx * y + x: (x, y) for y in range(Ly) for x in range(Lx)}

    hamiltonian = {}
    H2 = {}
    for bond in bonds:
        (x1, y1), (x2, y2), orientation = num2pt[bond[0]], num2pt[bond[1]], bond[2]
        Q1 = tensor_grid[y1][x1]
        Q2 = tensor_grid[y2][x2]

        if orientation == "H":
            QQ = ncon([Q1, Q2], [(-1, -2, 1, -5, -7), (1, -3, -4, -6, -8)])

        if orientation == "V":
            QQ = ncon([Q1, Q2], [(-1, -3, -4, 1, -7), (-2, 1, -5, -6, -8)])

        kernel = sp.linalg.null_space(QQ.reshape(-1, d**2))

        ham_term = 0.0
        for it in range(kernel.shape[1]):
            v = kernel[:, it]
            ham_term += ncon((np.conj(v), v), ([-1], [-2]))

        ham_term = ham_term.reshape((d, d, d, d))
        hamiltonian[bond] = ham_term

        H2[(y1, x1), (y2, x2)] = qu.qu(ham_term)

    quimb_hamiltonian = qtn.LocalHam2D(Lx, Ly, H2=H2)
    # quimb_hamiltonian.draw()

    return hamiltonian, quimb_hamiltonian


def apply_evolution_operator(parent_hamiltonian, peps, tau, Tmax, max_bond, phy_dim):
    d = phy_dim
    for bond, curr_term in parent_hamiltonian.terms.items():

        curr_gate = sp.linalg.expm(-curr_term.reshape(d**2, d**2) * 1j * tau * Tmax)
        peps.gate(
            curr_gate.reshape((d, d, d, d)), where=bond, contract="split", inplace=True
        )
        peps.compress_all(inplace=True, max_bond=max_bond)


def linear_interpolation(s, target_grid, initial_grid):
    tensor_grid = []
    for target_row, initial_row in zip(target_grid, initial_grid):
        tensor_row = []
        for target_tensor, initial_tensor in zip(target_row, initial_row):
            tensor = (1 - s) * initial_tensor + s * target_tensor
            tensor_row.append(tensor)
        tensor_grid.append(tensor_row)

    return tensor_grid


def adiabatic_state_preparation_2d(
    target_grid,
    initial_grid,
    bonds,
    Lx,
    Ly,
    phy_dim,
    Tmax,
    tau,
    max_bond,
    s_func,
    verbose=False,
):

    target_peps = qtn.PEPS(target_grid, shape="ldrup")
    target_peps.normalize(inplace=True)

    peps = qtn.PEPS(initial_grid, shape="ldrup")
    peps.normalize(inplace=True)

    ts = np.arange(0, Tmax + tau, tau)

    ss, target_fidelity, energy = {}, {}, {}
    for t in tqdm(ts):
        s = s_func(t)
        tensor_grid = linear_interpolation(s, target_grid, initial_grid)

        _, hamiltonian = construct_parent_hamiltonian(
            tensor_grid, bonds, Lx, Ly, phy_dim
        )
        apply_evolution_operator(hamiltonian, peps, tau, Tmax, max_bond, phy_dim)
        peps.normalize(inplace=True)

        ss[t] = s
        target_fidelity[t] = np.abs(peps.H @ target_peps)
        energy[t] = np.real(peps.compute_local_expectation(hamiltonian.terms))
        if verbose:
            print(
                f"\n{t=:.2f}, {s=:.5f}, e={energy[t]:.08f}, f={target_fidelity[t]:.08f}\n"
            )

    data = {"ss": ss, "target_fidelity": target_fidelity, "energy": energy}

    return data


def main():
    Lx, Ly = 2, 2
    Tmax, tau = 6, 0.04
    phy_dim = 16
    max_bond = 2

    # s_func = lambda t: np.sin( (np.pi/2)*np.sin( (np.pi/2)*t/Tmax )**2 )**2
    s_func = lambda t: np.sin((np.pi / 2) * t / Tmax) ** 2
    # s_func = lambda t: t/Tmax

    from misc_states import make_aklt_peps, make_bell_pair_peps

    target_grid, bonds = make_aklt_peps(Lx, Ly)
    initial_grid, _ = make_bell_pair_peps(Lx, Ly)

    data = adiabatic_state_preparation_2d(
        target_grid, initial_grid, bonds, Lx, Ly, phy_dim, Tmax, tau, max_bond, s_func
    )

    t_last = max(data["ss"].keys())
    s, e, f = (
        data["ss"][t_last],
        data["energy"][t_last],
        data["target_fidelity"][t_last],
    )
    print(f"\nfinal overlap is {s=:.5f}, e={e:.08f}, f={f:.08f}\n")


if __name__ == "__main__":
    main()
