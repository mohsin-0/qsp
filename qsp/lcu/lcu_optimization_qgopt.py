import numpy as np
import pickle as pkl


import quimb.tensor as qtn
import QGOpt as qgo
from tqdm import tqdm

# from tqdm.auto import tqdm
import tensorflow as tf


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
):

    lcu_list_var_c = list(map(qgo.manifolds.real_to_complex, lcu_list_var))
    # kappas = list(map(qgo.manifolds.real_to_complex, kappas))

    nomin = 0.0
    for it in range(no_of_layers):
        kappa = kappas[it]
        curr_mps = lcu_list_var_c[it * L : (it + 1) * L]
        curr_mps_shapes = lcu_shapes_list[it * L : (it + 1) * L]

        # with and without kappa, the optimal answer is same
        temp = mps_overlap(
            curr_mps, curr_mps_shapes, target_isometry_list, target_shapes_list
        )
        nomin = nomin + kappa * temp

    denomin = 0.0
    for it1 in range(no_of_layers):
        kappa1 = kappas[it1]
        curr_mps1 = lcu_list_var_c[it1 * L : (it1 + 1) * L]
        curr_mps_shapes1 = lcu_shapes_list[it1 * L : (it1 + 1) * L]

        for it2 in range(no_of_layers):
            kappa2 = kappas[it2]
            curr_mps2 = lcu_list_var_c[it2 * L : (it2 + 1) * L]
            curr_mps_shapes2 = lcu_shapes_list[it2 * L : (it2 + 1) * L]

            temp = mps_overlap(curr_mps1, curr_mps_shapes1, curr_mps2, curr_mps_shapes2)
            denomin = denomin + (kappa1 * kappa2) * temp

    overlap = tf.math.abs(nomin / tf.math.sqrt(denomin))
    overlap = -tf.cast(overlap, dtype=tf.float64)
    return overlap


def lcu_list_to_x(lcu_list, shape_structure):
    x = []
    for ten, shape in zip(lcu_list, shape_structure):
        for ele in ten.numpy().flatten():
            x.append(ele)
    x = np.array(x, dtype=np.float64)
    return x


def x_to_tf_lcu_list(x, shape_structure):
    xx = []
    offset = 0
    for shape in shape_structure:
        ten = tf.Variable(
            np.array(x[offset : (offset + np.prod(shape))], dtype=np.float64).reshape(
                shape
            ),
            dtype=tf.float64,
        )
        offset = offset + np.prod(shape)
        xx.append(ten)

    return xx


# @tf.function
def value_and_gradient(
    lcu_isometry_list_var,
    lcu_shapes_list,
    target_isometry_list,
    target_shapes_list,
    kappas,
    no_of_layers,
    L,
):

    with tf.GradientTape() as tape:
        overlap = compute_overlap(
            lcu_isometry_list_var,
            lcu_shapes_list,
            target_isometry_list,
            target_shapes_list,
            kappas,
            no_of_layers,
            L,
        )

    # grad = tape.gradient(overlap, lcu_isometry_list_var)
    grad = tape.gradient(overlap, lcu_isometry_list_var + [kappas])
    return overlap, grad


def lcu_unitary_circuit_optimization(
    target_mps, kappas, lcu_mps, max_iterations, verbose=False
):

    # tf.random.set_seed(42)

    iters = max_iterations  # number of iterations
    lr_i = 0.05  # initial learning rate
    lr_f = 0.001  # final learning rate

    # learning rate is multiplied by this coefficient each iteration
    decay = (lr_f / lr_i) ** (1 / iters)

    m = qgo.manifolds.StiefelManifold()  # complex Stiefel manifold
    riemannian_optimizer = qgo.optimizers.RAdam(m, learning_rate=lr_i)
    # riemannian_optimizer = tf.keras.optimizers.Adam()
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1.,
    #     decay_steps=10000,
    #     decay_rate=0.99)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    kappas = np.array(kappas, dtype=np.complex128)
    kappas = [kappa.tolist() for kappa in kappas]
    # kappas = list(map(qgo.manifolds.complex_to_real, kappas))
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
            lcu_shapes_list.append(shape)
            lcu_isometry_list.append(isometry)

    lcu_isometry_list = list(map(qgo.manifolds.complex_to_real, lcu_isometry_list))
    lcu_isometry_list_var = list(map(tf.Variable, lcu_isometry_list))
    shape_structure = [t.shape for t in lcu_isometry_list]

    no_of_layers, L = len(lcu_mps), target_mps.L
    overlaps_list = []
    for j in tqdm(range(iters)):
        # overlap, grad = value_and_gradient(lcu_isometry_list_var, kappas)
        overlap, grad = value_and_gradient(
            lcu_isometry_list_var,
            lcu_shapes_list,
            target_isometry_list,
            target_shapes_list,
            kappas,
            no_of_layers,
            L,
        )

        riemannian_optimizer.apply_gradients(zip(grad, lcu_isometry_list_var))
        riemannian_optimizer._set_hyper(
            "learning_rate", riemannian_optimizer._get_hyper("learning_rate") * decay
        )

        # vanilla optimization over kappas does not help
        # optimizer.apply_gradients(zip([grad[-1]], [kappas]))
        overlaps_list.append(overlap)
        grad_norm = np.linalg.norm(lcu_list_to_x(grad, shape_structure))
        if verbose:
            print(f" overlap={overlap.numpy():.06f}, \t |grad|={grad_norm:.06f},\t")

    tf.print(f"final overlap (after {iters} iters): ", overlaps_list[-1])

    ####
    lcu_mps_opt = convert_to_lcu_mps(
        lcu_isometry_list_var, lcu_shapes_list, len(lcu_mps), target_mps.L
    )
    data = {"lcu_mps_opt": lcu_mps_opt}
    return data


def convert_to_lcu_mps(tens_list, lcu_shapes_list, num_of_layers, L):
    tens = list(map(qgo.manifolds.real_to_complex, tens_list))
    tens = [tf.reshape(x, shape).numpy() for x, shape in zip(tens, lcu_shapes_list)]

    lcu_mps_opt = []
    for k_it in range(num_of_layers):
        curr_mps = [tens[k_it * L + site] for site in range(L)]
        curr_mps[0] = curr_mps[0].transpose((1, 0))
        curr_mps[-1] = curr_mps[-1]

        for it in range(1, L - 1):
            curr_mps[it] = curr_mps[it].transpose((0, 2, 1))

        lcu_mps_opt.append(qtn.MatrixProductState(curr_mps))

    return lcu_mps_opt


if __name__ == "__main__":
    with open("/home/mohsin/Documents/gh/gh_qsp/examples/temp_dump.pkl", "rb") as f:
        [target_mps, kappas, lcu_mps] = pkl.load(f)

    lcu_unitary_circuit_optimization(target_mps, kappas, lcu_mps)
