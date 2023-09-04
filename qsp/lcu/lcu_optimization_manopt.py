#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import tensorflow as tf


import pymanopt
from pymanopt.manifolds import ComplexGrassmann  # as Manifold
from pymanopt.optimizers import (
    ConjugateGradient,
)  # TrustRegions, ConjugateGradient, NelderMead
import quimb.tensor as qtn


from .lcu_optimization_misc import mps_overlap
from .lcu_optimization_misc import quimb_mps_to_tf_mps


@tf.function
def compute_overlap(
    lcu_list_var,
    lcu_shapes_list,
    target_isometry_list,
    target_shapes_list,
    kappas,
    no_of_layers,
    L,
    half_id,
    half_zr,
):

    nomin = 0.0
    for it in range(no_of_layers):
        kappa = kappas[it]
        curr_mps = [lcu_list_var[indx, :, :] for indx in range(it * L, (it + 1) * L)]

        curr_mps_shapes = lcu_shapes_list[it * L : (it + 1) * L]

        curr_mps[0] = half_id @ curr_mps[0]
        curr_mps[-1] = curr_mps[-1] @ half_zr

        temp = mps_overlap(
            curr_mps, curr_mps_shapes, target_isometry_list, target_shapes_list
        )
        # with and without kappa, the optimal answer is same
        nomin = nomin + kappa * temp

    denomin = 0.0
    for it1 in range(no_of_layers):
        kappa1 = kappas[it1]
        curr_mps1 = [lcu_list_var[indx, :, :] for indx in range(it1 * L, (it1 + 1) * L)]
        curr_mps_shapes1 = lcu_shapes_list[it1 * L : (it1 + 1) * L]

        curr_mps1[0] = half_id @ curr_mps1[0]
        curr_mps1[-1] = curr_mps1[-1] @ half_zr

        for it2 in range(no_of_layers):
            kappa2 = kappas[it2]
            curr_mps2 = [
                lcu_list_var[indx, :, :] for indx in range(it2 * L, (it2 + 1) * L)
            ]
            curr_mps_shapes2 = lcu_shapes_list[it2 * L : (it2 + 1) * L]

            curr_mps2[0] = half_id @ curr_mps2[0]
            curr_mps2[-1] = curr_mps2[-1] @ half_zr

            temp = mps_overlap(curr_mps1, curr_mps_shapes1, curr_mps2, curr_mps_shapes2)
            denomin = denomin + (kappa1 * kappa2) * temp

    overlap = tf.math.abs(nomin / tf.math.sqrt(denomin))
    overlap = -tf.cast(overlap, dtype=tf.float64)
    # print(1-overlap)

    return overlap


def lcu_unitary_circuit_optimization(
    target_mps, kappas, lcu_mps, max_time=7200, max_iterations=3000, verbose=False
):

    half_id = tf.convert_to_tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=tf.complex128
    )
    half_zr = tf.reshape(
        tf.convert_to_tensor([1.0, 0.0], dtype=tf.complex128), shape=(-1, 1)
    )

    kappas = np.array(kappas, dtype=np.complex128)
    kappas = [kappa.tolist() for kappa in kappas]
    kappas = tf.Variable(kappas)

    target_isometry_list, target_shapes_list = quimb_mps_to_tf_mps(
        target_mps, canonical_form="left"
    )
    target_isometry_list = list(map(tf.constant, target_isometry_list))
    target_isometry_list = list(
        map(lambda x: tf.cast(x, dtype=tf.complex128), target_isometry_list)
    )

    lcu_isometry_list = []
    lcu_shapes_list = []
    for mps in lcu_mps:
        isometry_list, shapes_list = quimb_mps_to_tf_mps(mps, canonical_form="left")
        for shape, isometry in zip(shapes_list, isometry_list):

            if isometry.shape == (4, 1):  # eng
                isometry = (
                    np.array(
                        [
                            isometry,
                            sp.linalg.null_space(isometry.T)[:, 0].reshape(-1, 1),
                        ]
                    )
                    .squeeze()
                    .T
                )

            elif isometry.shape == (2, 2):  # beginning
                temp = np.zeros((4, 2), dtype=np.complex128)
                temp[:2, :2] = isometry
                isometry = temp

            lcu_shapes_list.append(shape)
            lcu_isometry_list.append(isometry)

    no_of_layers, L = len(lcu_mps), target_mps.L

    tensor = np.zeros((no_of_layers * L, 4, 2), dtype=np.complex128)
    for it, isometry in enumerate(lcu_isometry_list):
        tensor[it, :, :] = isometry
    tensor = tf.convert_to_tensor(tensor, dtype=tf.complex128)

    manifold = ComplexGrassmann(4, 2, k=no_of_layers * L)
    euclidean_gradient = euclidean_hessian = None

    compute_overlap(
        tensor,
        lcu_shapes_list,
        target_isometry_list,
        target_shapes_list,
        kappas,
        no_of_layers,
        L,
        half_id,
        half_zr,
    )

    @pymanopt.function.tensorflow(manifold)
    def loss(x):
        return compute_overlap(
            x,
            lcu_shapes_list,
            target_isometry_list,
            target_shapes_list,
            kappas,
            no_of_layers,
            L,
            half_id,
            half_zr,
        )

    euclidean_gradient = euclidean_hessian = None

    problem = pymanopt.Problem(
        manifold,
        loss,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )

    # optimizer = TrustRegions(verbosity=2 * int(not quiet), max_time=7200)
    # estimated_spanning_set = optimizer.run( problem, Delta_bar=8 * np.sqrt(2), initial_point=tensor).point

    optimizer = ConjugateGradient(
        max_time=max_time, max_iterations=max_iterations, verbosity=int(verbose) * 2
    )
    estimated_spanning_set = optimizer.run(problem, initial_point=tensor).point

    lcu_mps_opt = convert_to_lcu_mps(
        estimated_spanning_set, len(kappas.numpy()), target_mps.L
    )
    data = {"lcu_mps_opt": lcu_mps_opt, "optimizer": optimizer}
    return data


def convert_to_lcu_mps(opt_tensor, num_of_layers, L):
    half_id = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    half_zr = np.array([1.0, 0.0])

    lcu_mps_opt = []
    for k_it in range(num_of_layers):
        curr_mps = [opt_tensor[k_it * L + site] for site in range(L)]
        curr_mps[0] = (half_id @ curr_mps[0]).reshape(2, 2).transpose((1, 0))
        curr_mps[-1] = (curr_mps[-1] @ half_zr).reshape(2, 2)

        for it in range(1, L - 1):
            curr_mps[it] = curr_mps[it].reshape((2, 2, 2)).transpose((0, 2, 1))

        lcu_mps_opt.append(qtn.MatrixProductState(curr_mps))

    return lcu_mps_opt


if __name__ == "__main__":
    pass
