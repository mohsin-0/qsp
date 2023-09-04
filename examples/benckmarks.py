#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl

import quimb.tensor as qtn


from qsp.misc_states import make_aklt_mps
from qsp.tsp import MPSPreparation
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

    data_seq = {}
    data_var_seq = {}
    data_qctn = {}
    data_lcu = {}
    data_var_lcu = {}

    mps_types = ["aklt", "random", "N2", "heisenberg"]
    for mps_type in mps_types:
        print(f"preparing {mps_type} state")
        target_mps = get_target_mps(mps_type)
        prep = MPSPreparation(target_mps)

        # ####### sequential mps preparation
        num_seq_layers = 2
        prep.sequential_unitary_circuit(num_seq_layers)
        data_seq[mps_type] = prep.seq_data

        # ####### variational mps preparation with sequential circicuit ansatz
        # data_var_seq[mps_type] = {}
        # for num_var_seq_layers in [2,3]:
        #     prep.sequential_unitary_circuit_optimization(num_var_seq_layers, max_iterations=4)
        #     data_var_seq[mps_type][num_var_seq_layers] = prep.var_seq_data

        ####### quantum circuit tensor network ansatz
        # data_qctn[mps_type] = {}
        # max_iterations = 4
        # for qctn_depth in [2,3]:#[2,4,6,8]:
        #     prep.quantum_circuit_tensor_network_ansatz(
        #         qctn_depth, max_iterations=max_iterations)
        #     data_qctn[mps_type][qctn_depth] = prep.qctn_data

        # ###### lcu mps preparation
        # num_lcu_layers = 4
        # prep.lcu_unitary_circuit(num_lcu_layers)
        # data_lcu[mps_type] = prep.lcu_data

        ###### variational mps preparation with lcu ansatz
        data_var_lcu[mps_type] = {}
        max_iterations = 4
        for num_var_lcu_layers in [2, 4]:
            prep.lcu_unitary_circuit_optimization(num_var_lcu_layers, max_iterations)
            data_var_lcu[mps_type][num_var_lcu_layers] = prep.var_lcu_data

    labels = {
        "aklt": "aklt state, $L=12$",
        "random": "random, $L=12$",
        "N2": "$N_2$, (STO-6G), $L=20$",
        "heisenberg": "HAF, $L=32$",
    }

    # for mps_type in mps_types:
    #     num_layers = data_seq[mps_type]['it_Ds']
    #     overlaps = data_seq[mps_type]['overlaps']
    #     plt.plot(num_layers, (1-np.abs(np.array(overlaps))), 'o-', label=labels[mps_type])
    # plt.xlabel('$num\ of\ seq.\ layers$')
    # plt.ylabel('$overlap\ with\ target\ wavefunction$')
    # plt.legend()

    # for mps_type in mps_types:
    #     num_layers = [2,3]
    #     overlaps = [-data_var_seq[mps_type][n]['tnopt'].loss_best for n in num_layers]
    #     plt.plot(num_layers, (1-np.abs(np.array(overlaps))), 'o-', label=labels[mps_type])
    # plt.title(r'$variational\ sequential\ unitary\ ansatz$')
    # plt.ylabel('$overlap\ with\ target\ wavefunction$')
    # plt.xlabel('$num\ of\ variational\ sequential\ layers$')
    # plt.legend()

    # for mps_type in mps_types:
    #     qctn_depth = [2,3]
    #     overlaps = [-data_qctn[mps_type][n]['tnopt'].loss_best for n in qctn_depth]
    #     plt.plot(qctn_depth, (1-np.abs(np.array(overlaps))), 'o-', label=labels[mps_type])
    # plt.title(r'$quantum\ circuit\ tensor\ network\ ansatz$')
    # plt.xlabel('$circuit\ depth$')
    # plt.ylabel('$overlap\ with\ target\ wavefunction$')
    # plt.legend()

    # for mps_type in mps_types:
    #     num_layers = data_lcu[mps_type]['it_Ds']
    #     overlaps = data_lcu[mps_type]['overlaps']
    #     plt.plot(num_layers, (1-np.abs(np.array(overlaps))), 'o-', label=labels[mps_type])
    # plt.title(r'$Linear\ combination\ of\ unitaries\ (LCU)$')
    # plt.ylabel('$overlap\ with\ target\ wavefunction$')
    # plt.xlabel('$num\ of\ lcu\ layers$')
    # plt.legend()

    for mps_type in mps_types:
        num_layers = [2, 4]
        overlaps = [data_var_lcu[mps_type][n]["overlap"] for n in num_layers]
        plt.plot(
            num_layers, (1 - np.abs(np.array(overlaps))), "o-", label=labels[mps_type]
        )
    plt.title(r"$Variational\ LCU$")
    plt.ylabel("$overlap\ with\ target\ wavefunction$")
    plt.xlabel("$num\ of\ lcu\ layers$")
    plt.legend()

    # ###### 1d adiabatic state preparation - random D=d=2 mps
    # prep_blocked = MPSPreparation(tensor_array_blocked, shape='lrp')
    # Tmax, tau = 8, 0.04 # total runtime, trotter step size
    # max_bond = 2
    # prep_blocked.adiabatic_state_preparation(Tmax, tau, max_bond, verbose=False)
