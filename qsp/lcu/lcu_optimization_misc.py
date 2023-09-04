#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensornetwork as tn

tn.set_default_backend("tensorflow")


def mps_to_isometry_list(tens_list, canonical_form):
    def reshape_rht2inds(A):
        shape = A.shape
        return (
            np.reshape(A, (shape[0], -1)) if len(shape) == 3 else np.reshape(A, (-1,))
        )

    def reshape_lft2inds(A):
        shape = A.shape
        return (
            np.reshape(A, (-1, shape[-1])) if len(shape) == 3 else np.reshape(A, (-1,))
        )

    L = len(tens_list)
    if canonical_form == "left":
        isometry_list = [
            reshape_lft2inds(A) if tid != 0 else A for tid, A in enumerate(tens_list)
        ]

    if canonical_form == "right":
        isometry_list = [
            reshape_rht2inds(A) if tid != (L - 1) else A
            for tid, A in enumerate(tens_list)
        ]

    return isometry_list


def canonical_form_sanity_check(tens_list, canonical_form):
    isometry_list = mps_to_isometry_list(tens_list, canonical_form)

    chks = []
    for u in isometry_list:
        shape = u.shape
        if len(shape) > 1:
            if shape[0] > shape[1]:
                chks.append(
                    np.allclose(
                        tf.eye(shape[1], dtype=tf.complex128), tf.linalg.adjoint(u) @ u
                    )
                )

            else:
                chks.append(
                    np.allclose(
                        tf.eye(shape[0], dtype=tf.complex128), u @ tf.linalg.adjoint(u)
                    )
                )

        else:
            chks.append(
                np.allclose(1.0, u[tf.newaxis] @ tf.linalg.adjoint(u[tf.newaxis]))
            )

    assert all(chks) == True, "every u in the list should be an isometry"


def quimb_mps_to_tf_mps(mps, canonical_form):
    mps.permute_arrays(shape="lpr")
    if canonical_form == "left":
        mps.left_canonize(normalize=True)

    elif canonical_form == "right":
        mps.right_canonize(normalize=True)

    tens_list = [mps.tensor_map[tid].data for tid in range(mps.L)]
    canonical_form_sanity_check(tens_list, canonical_form=canonical_form)

    isometry_list = mps_to_isometry_list(tens_list, canonical_form)
    shapes_list = [ten.shape for ten in tens_list]

    isometry_list = [
        isometry.reshape(-1, 1) if len(isometry.shape) == 1 else isometry
        for isometry in isometry_list
    ]

    return isometry_list, shapes_list


@tf.function
def mps_overlap(bra_isom, bra_shapes, ket_isom, ket_shapes):

    bra_mps = [tf.reshape(A, shape) for A, shape in zip(bra_isom, bra_shapes)]
    ket_mps = [
        tf.reshape(tf.math.conj(A), shape) for A, shape in zip(ket_isom, ket_shapes)
    ]

    L = len(bra_mps)
    co = 1
    list_of_tens = []
    list_of_inds = []
    for tid, (bra_A, ket_A) in enumerate(zip(bra_mps, ket_mps)):
        ket_A = tf.math.conj(ket_A)

        list_of_tens.append(bra_A)
        list_of_tens.append(ket_A)

        if tid == 0:
            list_of_inds.append([co, co + 1])
            list_of_inds.append([co, co + 2])
            co += 3

        elif tid == (L - 1):
            list_of_inds.append([co - 2, co])
            list_of_inds.append([co - 1, co])

        else:
            list_of_inds.append([co - 2, co, co + 1])
            list_of_inds.append([co - 1, co, co + 2])
            co += 3

    return tn.ncon(list_of_tens, list_of_inds)
