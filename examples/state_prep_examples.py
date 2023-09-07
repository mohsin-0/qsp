#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle as pkl

import quimb.tensor as qtn

from qsp.misc_states import make_aklt_mps
from qsp.misc_states import make_aklt_peps
from qsp.tsp import MPSPreparation, PEPSPreparation
from qsp.tsp_helper_routines import make_splitted_mps

def get_target_mps(mps_type):

    if mps_type == "aklt":
        tensor_array, _ = make_aklt_mps(L=12)
        tensor_array = make_splitted_mps(tensor_array)
        target_mps = qtn.MatrixProductState(tensor_array, shape="lrp")

    if mps_type == "random":
        target_mps = qtn.MPS_rand_state(L=12, bond_dim=4)
        target_mps.permute_arrays(shape="lrp")

    if mps_type in ["P4", "N2", "heisenberg"]:
        filenames = {
            "P4": "data/P4_6-31G_dist2.0000.pkl",
            "N2": "data/N2_STO-6G_dist2.0000.pkl",
            "heisenberg": "data/heisenberg_L32_dist0.8000.pkl",
        }
        with open(filenames[mps_type], "rb") as f:
            data = pkl.load(f)
        target_mps = data["quimb_mps"]

    return target_mps


if __name__ == "__main__":
    
    pass
    
    # TODO: git domain, setup file -> author, email, liscence
    # TODO: QCTN approximation of a PEPS
    # TODO:

    # # #######
    # target_mps = get_target_mps(mps_type='aklt')
    # prep = MPSPreparation(target_mps)
    # num_seq_layers = 6
    # seq_data = prep.sequential_unitary_circuit(num_seq_layers)

    # #######
    # target_mps = get_target_mps(mps_type='aklt')
    # prep = MPSPreparation(target_mps)
    # num_var_seq_layers = 2
    # var_seq_data = \
    #     prep.sequential_unitary_circuit_optimization(num_var_seq_layers, 
    #                                                  max_iterations=4)

    # #######
    # target_mps = get_target_mps(mps_type='aklt')
    # prep = MPSPreparation(target_mps)
    # qctn_depth = 8
    # qctn_data = prep.quantum_circuit_tensor_network_ansatz(qctn_depth, 
    #                                                        max_iterations=4)

    # #######
    # target_mps = get_target_mps(mps_type='N2')
    # prep = MPSPreparation(target_mps)
    # num_lcu_layers = 4
    # lcu_data = prep.lcu_unitary_circuit(num_lcu_layers)

    # ######
    # target_mps = get_target_mps(mps_type='N2')
    # num_var_lcu_layers = 4
    # data_var_lcu = prep.lcu_unitary_circuit_optimization(num_var_lcu_layers, 
    #                                                      max_iterations=4)

    # ####### 1d adiabatic state preparation - random D=d=2 mps
    # target_mps = qtn.MPS_rand_state(L=6, bond_dim=2)
    # prep = MPSPreparation(target_mps)
    # runtime, tau = 8, 0.04  # total runtime, trotter step size
    # max_bond = 2
    # adiabatic_data = prep.adiabatic_state_preparation(runtime, tau, max_bond)

    # ### 1d adiabatic state preparation - aklt
    # tensor_array, _ = make_aklt_mps(L=6)
    # prep = MPSPreparation(tensor_array, shape="lrp")
    # runtime, tau = 8, 0.04  # total runtime, trotter step size
    # max_bond = 2
    # prep.adiabatic_state_preparation(runtime, tau, max_bond, verbose=False)

    # # ## 2d adiabatic state preparation
    # Lx, Ly = 10, 4
    # target_grid, _ = make_aklt_peps(Lx, Ly)
    # prep = PEPSPreparation(target_grid)

    # Tmax, tau = 10, 0.04
    # max_bond = 2
    # prep.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)

    # plt.plot(prep.adiabatic_data['target_fidelity'].keys(),
    # prep.adiabatic_data['target_fidelity'].values(), '.-')
