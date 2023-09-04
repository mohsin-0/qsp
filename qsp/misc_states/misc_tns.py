#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

from ncon import ncon


pauli_x = [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]]

pauli_y = [[0.0 + 0.0j, -0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]]

pauli_z = [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]]

pauli_x = np.array(pauli_x)
pauli_y = np.array(pauli_y)
pauli_z = np.array(pauli_z)
I = pauli_z @ pauli_z


def make_bell_pair_mps(L, phys_dim):
    d, D = phys_dim, int(np.sqrt(phys_dim))

    bell_tensor = (np.eye(d)).reshape([D, D, d])

    bell_tens = [bell_tensor] * L
    bell_tens[0] = np.squeeze(bell_tens[0][0:1, :, :])
    bell_tens[-1] = np.squeeze(bell_tens[-1][:, 0:1, :])

    return bell_tens


def make_aklt_mps(L):
    s_ttl = {}
    for p, label in zip([pauli_x, pauli_y, pauli_z], ["x", "y", "z"]):
        tmp = 0
        tmp += ncon([p, I], [(-1, -3), (-2, -4)])
        tmp += ncon([I, p], [(-1, -3), (-2, -4)])
        s_ttl[label] = tmp.reshape(4, 4)

    s_sqr = s_ttl["x"] @ s_ttl["x"] + s_ttl["y"] @ s_ttl["y"] + s_ttl["z"] @ s_ttl["z"]
    s_sqr = s_sqr.reshape(4, 4)

    proj_symm = 0
    vals, vecs = sp.linalg.eigh(s_sqr)
    vecs = np.real(vecs)
    for vec_it, val in enumerate(vals):
        if np.allclose(val, 8.0):
            vec = vecs[:, vec_it]

            proj_symm += ncon([vec, np.conj(vec)], [(-1,), (-2,)])
            # s_val = (vec.T) @ s_sqr @ vec
            # z_val = (vec.T) @ s_ttl["z"] @ vec
            # print(s_val, z_val)
            # utils.print_vector(vec)
            # print('\n')

    proj_symm = proj_symm.reshape([2] * 2 + [4])
    singlet = np.sqrt(0.5) * np.array([[0.0, -1.0], [1.0, 0.0]])
    singlet_sqrt = sp.linalg.sqrtm(singlet)

    # aklt_tensor = ncon( [proj_symm, singlet], [(1, -2, -3), (-1, 1)] )
    aklt_tensor = ncon(
        [proj_symm, singlet_sqrt, singlet_sqrt], [(1, 2, -3), (-1, 1), (2, -2)]
    )

    # aklt_tensor = np.zeros([2,2, 2,2])
    # aklt_tensor[0,0, 0,0] = 1.

    # aklt_tensor[0,1, 0,1] = 1./(2)
    # aklt_tensor[0,1, 1,0] = 1./(2)

    # aklt_tensor[1,0, 0,1] = 1./(2)
    # aklt_tensor[1,0, 1,0] = 1./(2)

    # aklt_tensor[1,1, 1,1] = 1.
    # aklt_tensor = aklt_tensor.reshape((2,2,4))

    # aklt_tensor = ncon( [aklt_tensor, singlet], [(1, -2, -3), (-1, 1)] )

    ########
    isometry = np.zeros((2, 2, 3))
    isometry[0, 0, 0] = 1

    isometry[0, 1, 1] = 1.0 / np.sqrt(2)
    isometry[1, 0, 1] = 1.0 / np.sqrt(2)

    isometry[1, 1, 2] = 1
    isometry = isometry.reshape(4, 3)
    ########

    aklt_tens = [aklt_tensor] * L

    aklt_tens[0] = np.squeeze(aklt_tens[0][0, :, :])
    aklt_tens[-1] = np.squeeze(aklt_tens[-1][:, 0, :])

    onsite_isometries = [isometry] * L
    return aklt_tens, onsite_isometries


def make_bell_pair_peps(Lx, Ly):
    bell_tensor = (np.eye(16)).reshape((2, 2, 2, 2, 16))

    tensor_grid, bonds = construct_tensor_grid(bell_tensor, Lx, Ly)
    return tensor_grid, bonds


def make_aklt_peps(Lx, Ly):
    s_ttl = {}
    for p, label in zip([pauli_x, pauli_y, pauli_z], ["x", "y", "z"]):
        tmp = 0
        tmp += ncon([p, I, I, I], [(-1, -5), (-2, -6), (-3, -7), (-4, -8)])
        tmp += ncon([I, p, I, I], [(-1, -5), (-2, -6), (-3, -7), (-4, -8)])
        tmp += ncon([I, I, p, I], [(-1, -5), (-2, -6), (-3, -7), (-4, -8)])
        tmp += ncon([I, I, I, p], [(-1, -5), (-2, -6), (-3, -7), (-4, -8)])
        s_ttl[label] = tmp.reshape(16, 16)

    s_sqr = s_ttl["x"] @ s_ttl["x"] + s_ttl["y"] @ s_ttl["y"] + s_ttl["z"] @ s_ttl["z"]

    s_sqr = s_sqr.reshape(16, 16)

    proj_symm = 0
    vals, vecs = sp.linalg.eigh(s_sqr)
    vecs = np.real(vecs)
    for vec_it, val in enumerate(vals):
        if np.allclose(val, 24.0):
            vec = vecs[:, vec_it]

            proj_symm += ncon([vec, np.conj(vec)], [(-1,), (-2,)])
            # s_val = (vec.T) @ s_sqr @ vec
            # z_val = (vec.T) @ s_ttl["z"] @ vec
            # print(s_val, z_val)
            # utils.print_vector(vec)
            # print('\n')

    proj_symm = proj_symm.reshape([2] * 4 + [16])
    singlet = np.sqrt(0.5) * np.array([[0.0, -1.0], [1.0, 0.0]])

    aklt_tensor = ncon(
        (proj_symm, singlet, singlet), [(-1, -2, 3, 4, -5), (-3, 3), (-4, 4)]
    )

    tensor_grid, bonds = construct_tensor_grid(aklt_tensor, Lx, Ly)
    return tensor_grid, bonds


def construct_tensor_grid(local_tensor, Lx, Ly):
    pt2num = {(x, y): Lx * y + x for y in range(Ly) for x in range(Lx)}

    tensor_grid = []
    bonds = set()
    for y in range(Ly):
        tensor_row = []
        for x in range(Lx):

            if x < (Lx - 1):
                bonds.add((pt2num[(x, y)], pt2num[(x + 1, y)], "H"))

            if y < (Ly - 1):
                bonds.add((pt2num[(x, y)], pt2num[(x, y + 1)], "V"))

            tensor = local_tensor.copy()
            bdry = []
            if x == 0:
                tensor = ncon(
                    [tensor, np.array([1, 0]).reshape(2, -1)],
                    [(1, -2, -3, -4, -5), (1, -1)],
                )
                bdry.append("L")

            if x == (Lx - 1):
                tensor = ncon(
                    [tensor, np.array([1, 0]).reshape(2, -1)],
                    [(-1, -2, 3, -4, -5), (3, -3)],
                )
                bdry.append("R")

            if y == 0:
                tensor = ncon(
                    [tensor, np.array([1, 0]).reshape(2, -1)],
                    [(-1, 2, -3, -4, -5), (2, -2)],
                )
                bdry.append("B")

            if y == (Ly - 1):
                tensor = ncon(
                    [tensor, np.array([1, 0]).reshape(2, -1)],
                    [(-1, -2, -3, 4, -5), (4, -4)],
                )
                bdry.append("T")

            tensor_row.append(tensor)  # .squeeze())

            # print(f'({x},{y}), {bdry}')
        tensor_grid.append(tensor_row)

    return tensor_grid, bonds
