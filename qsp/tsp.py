#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import qiskit
import quimb.tensor as qtn

from .adiabatic import adiabatic_state_preparation_1d
from .adiabatic import adiabatic_state_preparation_2d

from .lcu import lcu_qgopt
from .lcu import lcu_manopt
from .lcu import lcu_unitary_circuit

from .misc_states import make_bell_pair_mps
from .misc_states import make_bell_pair_peps

from .q_circs import approximate_adiabatic_cost
from .q_circs import lcu_circuit_from_unitary_layers

from .sequential import apply_unitary_layers_on_wfn
from .sequential import quantum_circuit_tensor_network_ansatz
from .sequential import sequential_unitary_circuit_optimization
from .sequential import sequential_unitary_circuit

from .tsp_helper_routines import blockup_mps
from .tsp_helper_routines import cl_zero_mps
from .tsp_helper_routines import norm_mps_ovrlap


class MPSPreparation:
    """ontains implementations of different methods for preparing an MPS.

    Parameters
    ----------
    data : list(numpy.ndarray) or quimb.qtn.MatrixProductState.
        The input should be a list of three-index tensors (except the first
        and last, which are two-index tensors) specifying the MPS with open
        boundary conditions. The input could also be an instance of
        quimb.qtn.MatrixProductState class

    shape : str, option
        string of three letters (l, r, p) specifying the ordering of the
        indices in the MPS, where l = left, r = right, and p = physical


    Example usage
    --------
        bond_dim, phys_dim = 4, 2
        L=10
        tensor_array = [np.random.rand(bond_dim,bond_dim,phys_dim) for _ in range(L)]
        tensor_array[ 0] = np.random.rand(bond_dim,phys_dim)  # end points of mps
        tensor_array[-1] = np.random.rand(bond_dim,phys_dim)
        prep = MPSPreparation(tensor_array, shape='lrp')

    """

    def __init__(self, tensor_array, shape="lrp"):
        if isinstance(tensor_array, qtn.MatrixProductState):
            target_mps = tensor_array

        else:
            target_mps = qtn.MatrixProductState(tensor_array, shape=shape)
            target_mps.normalize()

        target_mps.normalize()
        target_mps.compress("right")

        self.phys_dim = target_mps.phys_dim()
        self.target_mps = target_mps
        self.shape = target_mps.shape
        self.L = target_mps.L

    def sequential_unitary_circuit(
        self, num_seq_layers, do_compression=False, max_bond_dim=None, verbose=False
    ):
        """The MPS is prepared as a sequence of unitaries, which are
        constructed using the disentangling algorithm described in
        https://arxiv.org/abs/1908.07958 by Ran et al.

        Parameters
        ----------
        num_seq_layers: int
            number of layers of sequential unitaries

        do_compression: bool, optional
            if True. the algirithm restricts the bond dimension of the
            disentangled mps at every step of the sequential unitary construction
            to max_bond_dim.

        max_bond_dim: int
            should be provided if do_compression is True.

        Returns
        -------
        dict
            contains the resulting circuit (as instance of qiskit.QuantumCircuit)
            and other useful information.
        """
        if self.phys_dim != 2:
            raise ValueError("only supports mps with physical dimesnion=2")

        if do_compression is True and max_bond_dim is None:
            raise ValueError(
                "since do_compression is True, max_bond_dim>0 should be specified"
            )

        print(
            f"preparing mps using sequential unitaries "
            f"(num_seq_layers={num_seq_layers})..."
        )

        data = sequential_unitary_circuit(
            self.target_mps,
            num_seq_layers,
            do_compression=do_compression,
            max_bond_dim=max_bond_dim,
            verbose=verbose,
        )

        self.seq_data = data
        unitaries, circ = data["unitaries"], data["circ"]

        # sanity check
        encoded_mps = apply_unitary_layers_on_wfn(unitaries, cl_zero_mps(self.L))
        encoded_mps.right_canonize(normalize=True)
        overlap = norm_mps_ovrlap(encoded_mps, self.target_mps)
        assert (
            np.abs(overlap - data["overlaps"][-1]) < 1e-12
        ), "overlap from seqential unitary construction does not match!"

        overlap_from_seq_circ = data["overlap_from_seq_circ"]
        temp_str = (
            ""
            if overlap_from_seq_circ is None
            else f" (from circ {overlap_from_seq_circ:0.8f})"
        )

        print(
            f"overlap from static seq. preparation = {np.abs(overlap):0.8f}{temp_str},\n"
            f"n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}\n"
        )

        return overlap, circ

    def sequential_unitary_circuit_optimization(
        self,
        num_var_seq_layers,
        do_compression=False,
        max_bond_dim=None,
        max_iterations=400,
        num_hops=1,
        verbose=False,
    ):
        """First, the MPS is prepared as a sequence of unitaries, which are
        constructed using the disentangling algorithm described in
        https://arxiv.org/abs/1908.07958 by Ran et al.
        Then, a variational optimization over unitaries is performed to further
        improve the overlap with the target MPS. Variational optimization is
        performed by 'scipy.optimize.basinhopping' via quimb package.

        Parameters
        ----------
        num_seq_layers: int
            number of layers of sequential unitaries

        do_compression: bool, optional
            if True. the algirithm restricts the bond dimension of the
            disentangled mps at every step to max_bond_dim.

        max_bond_dim: int
            should be provided if do_compression is True.

        max_iterations: int
            Maximum number of iterations for each optimization.

        num_hops: int
            Number of differently initialized optimizations

        Returns
        -------
        dict
            contains the resulting circuit (as instance of qiskit.QuantumCircuit)
            and other useful information.
        """

        if self.phys_dim != 2:
            raise ValueError("only supports mps with physical dimesnion=2")

        print(
            "doing variational optimization over sequential unitaries "
            f"(num_var_seq_layers={num_var_seq_layers})..."
        )

        self.var_seq_static_data = sequential_unitary_circuit(
            self.target_mps,
            num_var_seq_layers,
            do_compression=do_compression,
            max_bond_dim=max_bond_dim,
            verbose=verbose,
        )

        self.var_seq_data = sequential_unitary_circuit_optimization(
            self.target_mps,
            self.var_seq_static_data["unitaries"],
            max_iterations,
            num_hops,
        )

        circ, tnopt = self.var_seq_data["circ"], self.var_seq_data["tnopt"]

        overlap = -tnopt.loss_best
        overlap_from_seq_circ = self.var_seq_data["overlap_from_seq_circ"]
        temp_str = (
            ""
            if overlap_from_seq_circ is None
            else f" (from circ {overlap_from_seq_circ:0.8f})"
        )

        print(
            f"overllap after variational optimization = {overlap:0.8f}{temp_str},\n"
            f"n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}\n"
        )
        return overlap, circ

    def quantum_circuit_tensor_network_ansatz(
        self, qctn_depth, max_iterations=400, num_hops=1
    ):
        """The MPS is approximated by a Quantum Circuit Tensor Network (QCTN).
        QCTN consists of a layers single qubit (three parameter) unitaries
        followed by layers of CNOT gates. Variational optimization is
        performed by 'scipy.optimize.basinhopping' via quimb package.
        see: https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.011047

        Parameters
        ----------
        qctn_depth: int
            number of layers CNOT gates

        max_iterations: int
            Maximum number of iterations for each optimization.

        num_hops: int
            Number of differently initialized optimizations

        Returns
        -------
        dict
            contains the resulting circuit (as instance of qiskit.QuantumCircuit)
            and other useful information.
        """

        if self.phys_dim != 2:
            raise ValueError("only supports mps with physical dimesnion=2")

        print(
            "preparing mps using quantum circuit tensor network ansatz "
            f"(qctn_depth={qctn_depth})..."
        )

        self.qctn_data = quantum_circuit_tensor_network_ansatz(
            self.target_mps, qctn_depth, max_iterations, num_hops
        )

        circ, tnopt = self.qctn_data["circ"], self.qctn_data["tnopt"]

        overlap = -tnopt.loss_best
        overlap_from_seq_circ = self.qctn_data["overlap_from_seq_circ"]
        temp_str = (
            ""
            if overlap_from_seq_circ is None
            else f" (from circ {overlap_from_seq_circ:0.8f})"
        )

        print(
            f"overllap after qctn optimization ={overlap:0.8f}{temp_str},\n"
            f"n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}\n"
        )
        return overlap, circ
    

    def lcu_unitary_circuit(self, num_lcu_layers, verbose=False):
        """The MPS is approximated by linear combination of unitaries.
        Each of the unitary in the linear combination describes an MPS of bond
        dimension 2. The approximation algorithm is described in
        'https://arxiv.org/abs/2209.07106'

        Parameters
        ----------
        num_lcu_layers: int
            number of unitaries in the linear combinations

        Returns
        -------
        dict
            contains the resulting circuit (as instance of qiskit.QuantumCircuit)
            and other useful information.
        """
        if self.phys_dim != 2:
            raise ValueError("only supports mps with physical dimesnion=2")

            
        k = np.log2(num_lcu_layers)
        if (np.abs(k - int(k)) > 1e-12):
            raise ValueError(f'required num_var_lcu_layers={num_lcu_layers} '
                             'not a positive power of 2')

        print(
            f"preparing mps as linear combination of unitaries "
            f"(num_lcu_layers={num_lcu_layers})..."
        )
        data = lcu_unitary_circuit(self.target_mps, num_lcu_layers, verbose=verbose)
        self.lcu_data = data
        kappas, unitaries = data["kappas"], data["unitaries"]

        zero_wfn = cl_zero_mps(self.L)
        lcu_mps = [
            apply_unitary_layers_on_wfn(curr_us, zero_wfn) for curr_us in unitaries
        ]

        encoded_mps = cl_zero_mps(self.L) * 0
        for kappa, curr_mps in zip(kappas, lcu_mps):
            encoded_mps = encoded_mps + kappa * curr_mps

        encoded_mps.right_canonize(normalize=True)
        overlap = norm_mps_ovrlap(encoded_mps, self.target_mps)
        # assert np.abs(overlap-data['overlaps'][-1]) < 1e-14, f"overlap from lcu unitary does not match! {overlap}!={data['overlaps'][-1]}"

        k = int(np.ceil(np.log2(len(kappas))))
        L = self.L
        circ = qiskit.QuantumCircuit(L + k + 1)
        circ, overlap_from_lcu_circ = lcu_circuit_from_unitary_layers(
            circ, kappas, unitaries, self.target_mps
        )
        circ = qiskit.transpile(circ, basis_gates=["cx", "u3"])
        self.lcu_data["circ"] = circ

        temp_str = (
            ""
            if overlap_from_lcu_circ is None
            else f" (from circ {overlap_from_lcu_circ:0.8f})"
        )
        print(
            f"overllap after lcu. preparation = {np.abs(overlap):.8f}{temp_str}, ",
            f"n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}\n",
        )
        return overlap, circ

    def lcu_unitary_circuit_optimization(
        self, num_var_lcu_layers, max_iterations=500, verbose=False
    ):
        """First the MPS is approximated as a linear combination of unitaries
        by using the algorithm described in https://arxiv.org/abs/2209.07106.
        Then further variational optimization is performed over the MPS in the
        linear combination describing Grassmann manifold.

        Parameters
        ----------
        num_lcu_layers: int
            number of unitaries in the linear combinations

        max_iterations: int, optional
            maximum number of optimization steps.

        Returns
        -------
        dict
            contains the resulting circuit (as instance of qiskit.QuantumCircuit)
            and other useful information.
        """

        if self.phys_dim != 2:
            raise ValueError("only supports mps with physical dimesnion=2")


        k = np.log2(num_var_lcu_layers)
        if (np.abs(k - int(k)) > 1e-12):
            raise ValueError(f'required num_var_lcu_layers={num_var_lcu_layers} '
                             'not a positive power of 2')

        print("doing variational optimization over linear combination of "
              f"unitaries (num_var_lcu_layers={num_var_lcu_layers})...")

        self.var_lcu_static_data = lcu_unitary_circuit(
            self.target_mps, num_var_lcu_layers, verbose=verbose
        )

        kappas = self.var_lcu_static_data["kappas"]
        unitaries = self.var_lcu_static_data["unitaries"]

        k = int(np.ceil(np.log2(len(kappas))))
        L = self.L
        circ = qiskit.QuantumCircuit(L + k + 1)
        circ, overlap_from_lcu_circ = lcu_circuit_from_unitary_layers(
            circ, kappas, unitaries, self.target_mps
        )
        circ = qiskit.transpile(circ, basis_gates=["cx", "u3"])

        temp_str = (
            ""
            if overlap_from_lcu_circ is None
            else f" (from circ {np.abs(overlap_from_lcu_circ):0.8f})"
        )
        print(
            "overlap before lcu optimization = "
            f'{np.abs(self.var_lcu_static_data["overlaps"][-1]):.10f} {temp_str}, '
            f"n_gates={circ.size()}, n_2qg={circ.num_nonlocal_gates()}"
        )

        lcu_mps = [
            apply_unitary_layers_on_wfn(curr_us, cl_zero_mps(self.L))
            for curr_us in unitaries
        ]

        method_name = ""
        if all([D == 2 for mps in lcu_mps for D in mps.bond_sizes()]):
            method_name = "manopt"
            self.var_lcu_data = lcu_manopt(
                self.target_mps,
                kappas,
                lcu_mps,
                max_iterations=max_iterations,
                verbose=verbose,
            )

            lcu_mps_opt = self.var_lcu_data["lcu_mps_opt"]
            kappas = self.var_lcu_static_data["kappas"]

        else:
            method_name = "qgopt"
            self.var_lcu_data = lcu_qgopt(
                self.target_mps,
                kappas,
                lcu_mps,
                max_iterations=max_iterations,
                verbose=verbose,
            )
            lcu_mps_opt = self.var_lcu_data["lcu_mps_opt"]
            kappas = self.var_lcu_static_data["kappas"]

        self.var_lcu_data["method_name"] = method_name

        encoded_mps = cl_zero_mps(self.L) * 0
        for kappa, curr_mps in zip(kappas, lcu_mps_opt):
            encoded_mps = encoded_mps + kappa * curr_mps
        encoded_mps.right_canonize(normalize=True)
        overlap = norm_mps_ovrlap(encoded_mps, self.target_mps)

        k = int(np.ceil(np.log2(len(kappas))))
        L = self.L
        circ = qiskit.QuantumCircuit(L + k + 1)
        circ, overlap_from_lcu_circ = lcu_circuit_from_unitary_layers(
            circ, kappas, unitaries, self.target_mps
        )
        circ = qiskit.transpile(circ, basis_gates=["cx", "u3"])
        self.var_lcu_data["circ"] = circ
        self.var_lcu_data["overlap"] = overlap

        print("overllap after lcu optimization "
              f"({method_name}) = {np.abs(overlap):.8f}\n")
        return overlap, circ

    def adiabatic_state_preparation(self, runtime, tau, max_bond_dim, verbose=False):
        """performs adiabatic preparation of the mps using the algorithm
        described in https://arxiv.org/abs/2209.01230

        Parameters
        ----------
        runtime: float
            total runtime for adiabatic preparation

        tau: float
            trotter step size

        max_bond_dim: int
            max. bond dimension of the mps during adiabatic evolution

        Returns
        -------
        dict
            contains fidelity of the adiabatically prepared state with respect
            to the target mps and approximate gate cost of the adiabatic
            algorithm.
        """

        print(
            "adiabatic state preparation of mps:\n"
            f"runtime={runtime}, tau={tau:0.04}, steps={int(runtime/tau)}, max_bond_dim={max_bond_dim}"
        )

        Ds = self.target_mps.bond_sizes()
        D, d = max(Ds), self.target_mps.phys_dim()

        # assumes uniform bond dimension
        assert all(i == Ds[0] for i in Ds)

        if D**2 > d:
            print("given mps is not injective. blocking it now ...")
            # block the to be in injective form
            block_size = int(np.ceil(2 * np.log(D) / np.log(d)))
            blocked_mps = blockup_mps(self.target_mps, block_size)
        else:
            blocked_mps = self.target_mps

        s_func = (
            lambda t: np.sin((np.pi / 2) * np.sin((np.pi / 2) * t / runtime) ** 2) ** 2
        )
        # s_func = lambda t: np.sin( (np.pi/2)*t/runtime)**2
        # s_func = lambda t: t/runtime

        # # ####################################
        initial_tens = make_bell_pair_mps(
            L=blocked_mps.L, phys_dim=blocked_mps.phys_dim()
        )
        initial_mps = qtn.MatrixProductState(initial_tens, shape="lrp")

        data = adiabatic_state_preparation_1d(
            blocked_mps,
            initial_mps,
            runtime,
            tau,
            s_func,
            max_bond_dim,
            verbose=verbose,
        )

        self.adiabatic_data = data

        gates = [gate for gate_layer in data["gates"].values() for gate in gate_layer]
        n_gates, n_2qg = approximate_adiabatic_cost(gates)

        t_last = max(data["ss"].keys())
        s, e = data["ss"][t_last], data["energy"][t_last]
        curr_f, tar_f = (
            data["current_fidelity"][t_last],
            data["target_fidelity"][t_last],
        )
        print(
            f"final overlap @ {s=:.5f} is e={e:.08f}, "
            f"curr_f={curr_f:.08f}, target_fidelity={tar_f:.08f}\n"
            f"approximate n_gates={int(n_gates)}, and n_2qg={int(n_2qg)}\n"
        )
        return tar_f, n_2qg


class PEPSPreparation:
    def __init__(self, tensor_grid, shape="ldrup"):
        self.target_grid = tensor_grid
        self.shape = shape

        self.Lx, self.Ly = len(tensor_grid[0]), len(tensor_grid)
        self.phy_dim = tensor_grid[0][0].shape[-1]

    def adiabatic_state_preparation(self, Tmax, tau, max_bond, verbose=False):
        """performs adiabatic preparation of the peps using the algorithm
        described in https://arxiv.org/abs/2209.01230

        Parameters
        ----------
        runtime: float
            total runtime for adiabatic preparation

        tau: float
            trotter step size

        max_bond_dim: int
            max. bond dimension of the mps during adiabatic evolution

        Returns
        -------
        dict
            contains fidelity of the adiabatically prepared state with respect
            to the target mps and approximate gate cost of the adiabatic
            algorithm.
        """

        print("adiabatic state preparation of peps:\n"
            f"runtime={Tmax}, tau={tau:0.04}, steps={int(Tmax/tau)}, "
            f"max_bond={max_bond}")

        # s_func = lambda t: np.sin( (np.pi/2)*np.sin( (np.pi/2)*t/Tmax )**2 )**2
        s_func = lambda t: np.sin((np.pi / 2) * t / Tmax) ** 2
        # s_func = lambda t: t/Tmax

        # target_grid, bonds = make_aklt_peps(Lx, Ly)
        initial_grid, bonds = make_bell_pair_peps(self.Lx, self.Ly)

        data = adiabatic_state_preparation_2d(
            self.target_grid,
            initial_grid,
            bonds,
            self.Lx,
            self.Ly,
            self.phy_dim,
            Tmax,
            tau,
            max_bond,
            s_func,
            verbose=verbose,
        )
        self.adiabatic_data = data

        t_last = max(data["ss"].keys())
        s, e, f = (
            data["ss"][t_last],
            data["energy"][t_last],
            data["target_fidelity"][t_last],
        )
        print(f"\n2d adiabatic preparation: @ {s=:.5f}, e={e:.08f}, fidelity={f:.08f}")
