import quimb.tensor as qtn
import pickle as pkl

import numpy as np
import tensorflow as tf
import scipy.optimize as sopt    

import QGOpt as qgo
import tensornetwork as tn

import matplotlib.pyplot as plt
from tqdm import tqdm


def mps_to_isometry_list(tens_list, canonical_form): 
    def reshape_rht2inds(A):
        shape = A.shape
        return np.reshape(A,(shape[0],-1)) if len(shape)==3 else np.reshape(A,(-1,))
        
    def reshape_lft2inds(A):
        shape = A.shape
        return np.reshape(A,(-1,shape[-1])) if len(shape)==3 else np.reshape(A,(-1,))
     
    L = len(tens_list) 
    if canonical_form=='left':
        isometry_list = [reshape_lft2inds(A) if tid!=0  else A for tid, A in enumerate(tens_list)]
        
    if canonical_form=='right':
        isometry_list = [reshape_rht2inds(A) if tid!=(L-1) else A for tid, A in enumerate(tens_list)]
        
    return isometry_list
    

def canonical_form_sanity_check(tens_list, canonical_form):
    isometry_list = mps_to_isometry_list(tens_list, canonical_form)
    
    chks = []
    for u in isometry_list:
        shape = u.shape
        if len(shape)>1: 
            if shape[0]>shape[1]:
                chks.append(np.allclose(tf.eye(shape[1], dtype=tf.complex128), tf.linalg.adjoint(u)@u))
                
            else:
                chks.append(np.allclose(tf.eye(shape[0], dtype=tf.complex128), u@tf.linalg.adjoint(u)))
                
        else:
            chks.append(np.allclose(1., u[tf.newaxis]@tf.linalg.adjoint(u[tf.newaxis])))
            
    assert all(chks)==True, 'every u in the list should be an isometry' 


def quimb_mps_to_tf_mps(mps, canonical_form):
    mps.permute_arrays(shape='lpr')
    if canonical_form=='left':
        mps.left_canonize(normalize=True)
        
    elif canonical_form=='right':
        mps.right_canonize(normalize=True)
        
    tens_list = [mps.tensor_map[tid].data for tid in range(mps.L)]
    canonical_form_sanity_check(tens_list, canonical_form=canonical_form)
    
    isometry_list = mps_to_isometry_list(tens_list, canonical_form)
    shapes_list = [ten.shape for ten in tens_list]

    isometry_list = [isometry.reshape(-1,1) if len(isometry.shape)==1 else isometry  for isometry in isometry_list]
    
    return isometry_list, shapes_list


# @tf.function
def mps_overlap(bra_isom, bra_shapes, ket_isom, ket_shapes):
    
    # overlap = 0.
    # for u in bra_isom:
    #     overlap = overlap + tf.linalg.trace(tf.linalg.adjoint(u)@u)
    # return overlap

    bra_mps = [tf.reshape(A, shape) for A, shape in zip(bra_isom, bra_shapes)]
    ket_mps = [tf.reshape(tf.math.conj(A), shape) for A, shape in zip(ket_isom, ket_shapes)]
    
    
    L = len(bra_mps)
    co = 1
    list_of_tens = []
    list_of_inds = []
    for tid, (bra_A, ket_A) in enumerate(zip(bra_mps, ket_mps)):
        ket_A = tf.math.conj(ket_A)
        
        list_of_tens.append(bra_A)
        list_of_tens.append(ket_A)
        
        if tid==0:    
            list_of_inds.append([co, co+1])
            list_of_inds.append([co, co+2])
            co += 3 
            
        elif tid==(L-1):
            list_of_inds.append([co-2, co])
            list_of_inds.append([co-1, co])
            
        else:
            list_of_inds.append([co-2, co, co+1])
            list_of_inds.append([co-1, co, co+2])
            co += 3 
            
    return tn.ncon(list_of_tens, list_of_inds)


# @tf.function
def compute_overlap(lcu_list_var, lcu_shapes_list, 
                    target_isometry_list, target_shapes_list, 
                    kappas, no_of_layers, L):
    
    lcu_list_var_c = list(map(qgo.manifolds.real_to_complex, lcu_list_var))
    kappas = list(map(qgo.manifolds.real_to_complex, kappas))
    
    nomin = 0.    
    for it in range(no_of_layers):
        kappa  = kappas[it]
        curr_mps = lcu_list_var_c[it*L:(it+1)*L]
        curr_mps_shapes = lcu_shapes_list[it*L:(it+1)*L]
        
        # with and without kappa, the optimal answer is same
        nomin = nomin + kappa * mps_overlap(curr_mps, curr_mps_shapes,
                                            target_isometry_list, target_shapes_list)
                    
        # nomin = nomin + mps_overlap(curr_mps, curr_mps_shapes,
        #                             target_isometry_list, target_shapes_list)

        
    denomin = 0.    
    for it1 in range(no_of_layers):
        kappa1  = kappas[it1]
        curr_mps1 = lcu_list_var_c[it1*L:(it1+1)*L]
        curr_mps_shapes1 = lcu_shapes_list[it1*L:(it1+1)*L]
        
        for it2 in range(no_of_layers):
            kappa2  = kappas[it2]
            curr_mps2 = lcu_list_var_c[it2*L:(it2+1)*L]
            curr_mps_shapes2 = lcu_shapes_list[it2*L:(it2+1)*L]
            
            denomin = denomin + (kappa1 * kappa2) * mps_overlap(curr_mps1, curr_mps_shapes1,
                                                                curr_mps2, curr_mps_shapes2)
                                                                
            # denomin = denomin + mps_overlap(curr_mps1, curr_mps_shapes1,
            #                                 curr_mps2, curr_mps_shapes2)
        
    
        
    overlap = tf.math.abs(nomin/tf.math.sqrt(denomin))
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
        ten = tf.Variable(np.array(x[offset:(offset+np.prod(shape))], dtype=np.float64).reshape(shape),dtype=tf.float64)
        offset = offset+np.prod(shape)
        xx.append(ten)
    
    return xx


# @tf.function
def value_and_gradient(lcu_isometry_list_var, lcu_shapes_list, 
                       target_isometry_list, target_shapes_list, 
                       kappas, no_of_layers, L):
    
    with tf.GradientTape() as tape:
        overlap = compute_overlap(lcu_isometry_list_var, lcu_shapes_list, 
                                  target_isometry_list, target_shapes_list, 
                                  kappas, no_of_layers, L)
        
    grad = tape.gradient(overlap, lcu_isometry_list_var)
    # grad = tape.gradient(overlap, lcu_isometry_list_var+[kappas])
    return overlap, grad


def lcu_unitary_circuit_optimization(target_mps, kappas, lcu_mps): 
    tn.set_default_backend("tensorflow")
    tf.random.set_seed(42)
    
    iters = 500 # number of iterations
    lr_i = 0.05 # initial learning rate
    lr_f = 0.05 # final learning rate
    
    # learning rate is multiplied by this coefficient each iteration
    decay = (lr_f / lr_i) ** (1 / iters)
    
    m = qgo.manifolds.StiefelManifold()  # complex Stiefel manifold
    riemannian_optimizer = qgo.optimizers.RAdam(m, learning_rate=0.1)
    # optimizer = tf.keras.optimizers.Adam()
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1.,
    #     decay_steps=10000,
    #     decay_rate=0.99)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    
    kappas = np.array(kappas, dtype=np.complex128)
    kappas = [kappa.tolist() for kappa in kappas]
    kappas = list(map(qgo.manifolds.complex_to_real, kappas)) 
    kappas = tf.Variable(kappas)

    target_isometry_list, target_shapes_list = quimb_mps_to_tf_mps(target_mps, canonical_form='left')
    target_isometry_list = list(map(tf.constant, target_isometry_list)) 
    target_isometry_list = list(map(lambda x: tf.cast(x, dtype=tf.complex128), target_isometry_list))
    
    lcu_isometry_list = []
    lcu_shapes_list = []
    for mps in lcu_mps:
        isometry_list, shapes_list = quimb_mps_to_tf_mps(mps, canonical_form='left')
        
        for shape, isometry in zip(shapes_list, isometry_list):
            lcu_shapes_list.append(shape)
            lcu_isometry_list.append(isometry)
    

    lcu_isometry_list = list(map(qgo.manifolds.complex_to_real, lcu_isometry_list)) 
    lcu_isometry_list_var = list(map(tf.Variable, lcu_isometry_list))
    shape_structure = [t.shape for t in lcu_isometry_list]

    # the optimization is not good as it is not over the stieffel manifold
    # def func(x):
    #     lcu_list = x_to_tf_lcu_list(x, shape_structure)
    #     overlap, grad = value_and_gradient(lcu_list, kappas)        
    #     grad = lcu_list_to_x(lcu_isometry_list_var, shape_structure).reshape(-1,1)
    #     print('..')
    #     return overlap.numpy().astype(np.float64), grad
    
    # resdd= sopt.minimize(fun=func,  
    #                       x0=lcu_list_to_x(lcu_isometry_list_var, shape_structure),
    #                       jac=True, 
    #                       method='L-BFGS-B', 
    #                       options={'disp': True})
    
    no_of_layers, L = len(lcu_mps), target_mps.L
    
    overlaps_list = []
    for j in tqdm(range(iters)):        
        # overlap, grad = value_and_gradient(lcu_isometry_list_var, kappas)
        overlap, grad = value_and_gradient(lcu_isometry_list_var, lcu_shapes_list, 
                                           target_isometry_list, target_shapes_list, 
                                           kappas, no_of_layers, L)
        
        
        riemannian_optimizer.apply_gradients(zip(grad, lcu_isometry_list_var))    
        
        learning_rate = riemannian_optimizer._get_hyper("learning_rate") * decay
        riemannian_optimizer._set_hyper("learning_rate", learning_rate)
        
        # vanilla optimization over kappas does not help
        # optimizer.apply_gradients(zip([grad[-1]], [kappas]))        
        overlaps_list.append(overlap)
        grad_norm = np.linalg.norm(lcu_list_to_x(grad, shape_structure))
        print(f' {overlap.numpy()=},\t{grad_norm=},\t')
        
    tf.print(f'final overlap (after {iters} iters): ', overlaps_list[-1])
    
    
if __name__ == "__main__":
    pass 