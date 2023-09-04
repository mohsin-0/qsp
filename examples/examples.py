#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl

import quimb.tensor as qtn

from qsp.misc_states import make_aklt_mps
from qsp.misc_states import make_aklt_peps
from qsp.tsp import MPSPreparation, PEPSPreparation
from qsp.tsp_helper_routines import make_splitted_mps


if __name__ == "__main__":
    # TODO: git,  add references + add documentation + comments + readme (installation comments)
    # TODO: Notebooks - one example notebook. one notebook for benchmarks
    # TODO: publish
    # TODO: fix verbose
    # TODO: ordering of input arguments and type checking
    # TODO: git domain, setup file -> author, email, liscence
    # TODO: basinhopping overlap from circuit don't match if the n_iters are isufficient
    # change to a different method
    # more instructive output - suggent to increase the number of iterations
    # TODO: QCTN approximation of a PEPS

    mps_type = "aklt"  #'N2'#'P4'#'aklt'#'heisenberg'#'random'
    if mps_type == "aklt":
        tensor_array, _ = make_aklt_mps(L=4)
        tensor_array = make_splitted_mps(tensor_array)
        target_mps = qtn.MatrixProductState(tensor_array, shape="lrp")

    if mps_type == "random":
        target_mps = qtn.MPS_rand_state(L=12, bond_dim=4)
        target_mps.permute_arrays(shape="lrp")

    if mps_type in ["P4", "N2", "heisenberg"]:
        filenames = {
            "P4": "data/P4_6-31G_dist2.0000.pkl",  #
            "N2": "data/N2_STO-6G_dist2.0000.pkl",
            "heisenberg": "data/heisenberg_L32_dist0.8000.pkl",
        }
        with open(filenames[mps_type], "rb") as f:
            data = pkl.load(f)
        target_mps = data["quimb_mps"]

    # #######
    # prep = MPSPreparation(target_mps)
    # number_of_layers = 4
    # prep.sequential_unitary_circuit(number_of_layers,
    #                                 do_compression=False,
    #                                 verbose=False)

    # #######
    # prep = MPSPreparation(target_mps)
    # number_of_layers = 2
    # n_iter, nhop = 400, 4
    # prep.sequential_unitary_circuit_optimization(number_of_layers,
    #                                              do_compression=False,
    #                                              n_iter=n_iter, nhop=nhop)

    # #######
    # prep = MPSPreparation(target_mps)
    # depth = 4
    # n_iter, nhop = 2, 2,
    # prep.quantum_circuit_tensor_network_ansatz(depth, n_iter=n_iter, nhop=nhop)

    #######
    # prep = MPSPreparation(target_mps)
    # number_of_lcu_layers = 4
    # prep.lcu_unitary_circuit(number_of_lcu_layers, verbose=False)

    ######
    prep = MPSPreparation(target_mps)
    number_of_lcu_layers = 4
    max_iterations = 4
    prep.lcu_unitary_circuit_optimization(
        number_of_lcu_layers, max_iterations, verbose=False
    )

    ####### 1d adiabatic state preparation - random D=d=2 mps
    # target_mps = qtn.MPS_rand_state(L=6, bond_dim=2)
    # prep = MPSPreparation(target_mps)
    # Tmax, tau = 8, 0.04 #total runtime, trotter step size
    # max_bond = 2
    # prep.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)
    # plt.plot(prep.adiabatic_data['target_fidelity'].keys(),
    # prep.adiabatic_data['target_fidelity'].values(), '.-')

    ### 1d adiabatic state preparation - aklt
    # tensor_array, _ = make_aklt_mps(L=6)
    # prep = MPSPreparation(tensor_array, shape='lrp')
    # Tmax, tau = 8, 0.04 # total runtime, trotter step size
    # max_bond = 2
    # prep.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)
    # plt.plot(prep.adiabatic_data['target_fidelity'].keys(),
    # prep.adiabatic_data['target_fidelity'].values(), '.-')

    # ## 2d adiabatic state preparation
    # Lx, Ly = 2, 2
    # target_grid, _ = make_aklt_peps(Lx, Ly)
    # prep = PEPSPreparation(target_grid)

    # Tmax, tau = 4, 0.04
    # max_bond = 2
    # prep.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)

    # plt.plot(prep.adiabatic_data['target_fidelity'].keys(),
    # prep.adiabatic_data['target_fidelity'].values(), '.-')
