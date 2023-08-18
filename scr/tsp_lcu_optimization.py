#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import tensorflow as tf 

import pymanopt
from pymanopt.manifolds import ComplexGrassmann as Manifold
from pymanopt.optimizers import TrustRegions, ConjugateGradient, NelderMead
import tensornetwork as tn

tn.set_default_backend("tensorflow")
tf.random.set_seed(42)

NPTYPE = np.complex128
TFTYPE = tf.complex128


def mps_to_isometry_list(tens_list, canonical_form): 
    def reshape_rht2inds(A):
        shape = A.shape
        return np.reshape(A,(shape[0],-1)) if len(shape)==3 else np.reshape(A,(-1,))
        
    def reshape_lft2inds(A):
        shape = A.shape
        return np.reshape(A,(-1,shape[-1])) if len(shape)==3 else np.reshape(A,(-1,))
     
    L = len(tens_list) 
    if canonical_form=='left':
        isometry_list = [reshape_lft2inds(A) if tid!=0  else A 
                         for tid, A in enumerate(tens_list)]
        
    if canonical_form=='right':
        isometry_list = [reshape_rht2inds(A) if tid!=(L-1) else A 
                         for tid, A in enumerate(tens_list)]
        
    return isometry_list
    

def canonical_form_sanity_check(tens_list, canonical_form):
    isometry_list = mps_to_isometry_list(tens_list, canonical_form)
    
    chks = []
    for u in isometry_list:
        shape = u.shape
        if len(shape)>1: 
            if shape[0]>shape[1]:
                chks.append(np.allclose(tf.eye(shape[1], dtype=TFTYPE), 
                                        tf.linalg.adjoint(u)@u))
                
            else:
                chks.append(np.allclose(tf.eye(shape[0], dtype=TFTYPE), 
                                        u@tf.linalg.adjoint(u)))
                
        else:
            chks.append(np.allclose(1., 
                                    u[tf.newaxis]@tf.linalg.adjoint(u[tf.newaxis])))
            
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

    isometry_list = [isometry.reshape(-1,1) if len(isometry.shape)==1 else isometry  
                     for isometry in isometry_list]
    
    return isometry_list, shapes_list


# @tf.function
def mps_overlap(bra_isom, bra_shapes, ket_isom, ket_shapes):
    bra_mps = [trim_reshape(A, shape) 
               for A, shape in zip(bra_isom, bra_shapes)]
    
    ket_mps = [trim_reshape(tf.math.conj(A), shape) 
               for A, shape in zip(ket_isom, ket_shapes)]
    
    # bra_mps = [tf.reshape(A, shape) 
    #            for A, shape in zip(bra_isom, bra_shapes)]
    
    # ket_mps = [tf.reshape(tf.math.conj(A), shape) 
    #            for A, shape in zip(ket_isom, ket_shapes)]
    
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


def trim_reshape(ten, new_shape):
    half_zr = tf.reshape(tf.convert_to_tensor([1., 0.], dtype=TFTYPE), shape=(-1,1))
    
    # if ten.shape==(2,2) and new_shape==(2,1):
    #     return tf.reshape(ten@half_zr, new_shape)
    
    # if ten.shape==(4,2) and new_shape==(1,2,1):
    #     return tf.reshape(tf.transpose(half_zr,(1,0))@tf.reshape(ten@half_zr, [2,2]), new_shape) 
    
    # if ten.shape==(4,2) and new_shape==(1,2,2):
    #     return tf.reshape(tf.transpose(half_zr,(1,0))@tf.reshape(ten, [2,4]), new_shape)
        
    # if ten.shape==(4,2) and new_shape==(2,2,1):
    #     return tf.reshape(tf.reshape(ten, [2,2,2])@half_zr, new_shape)
            
    
    # if ten.shape==(4,1) and new_shape==(1,2):
    #     return tf.reshape(tf.transpose(half_zr,(1,0))@tf.reshape(ten, [2,2]), new_shape)
        
    
    return tf.reshape(ten, new_shape)
        


# @tf.function
def compute_overlap(lcu_list_var, lcu_shapes_list, 
                    target_isometry_list, target_shapes_list, 
                    kappas, 
                    no_of_layers, L, 
                    half_id, half_zr):

    nomin = 0.    
    for it in range(no_of_layers):
        kappa  = kappas[it]
        curr_mps = [lcu_list_var[indx,:,:] for indx in range(it*L,(it+1)*L)]

        curr_mps_shapes = lcu_shapes_list[it*L:(it+1)*L]
        
                
        curr_mps[ 0] = half_id@curr_mps[ 0]
        curr_mps[-1] = curr_mps[-1]@half_zr
        
        # with and without kappa, the optimal answer is same
        nomin = nomin + kappa * mps_overlap(curr_mps, 
                                            curr_mps_shapes,
                                            target_isometry_list, 
                                            target_shapes_list)


    denomin = 0.    
    for it1 in range(no_of_layers):
        kappa1  = kappas[it1]
        curr_mps1 = [lcu_list_var[indx,:,:] for indx in range(it1*L,(it1+1)*L)]
        curr_mps_shapes1 = lcu_shapes_list[it1*L:(it1+1)*L]
        
        curr_mps1[ 0] = half_id@curr_mps1[ 0]
        curr_mps1[-1] = curr_mps1[-1]@half_zr
        
        for it2 in range(no_of_layers):
            kappa2  = kappas[it2]
            curr_mps2 = [lcu_list_var[indx,:,:] for indx in range(it2*L,(it2+1)*L)]
            curr_mps_shapes2 = lcu_shapes_list[it2*L:(it2+1)*L]
            
            curr_mps2[ 0] = half_id@curr_mps2[ 0]
            curr_mps2[-1] = curr_mps2[-1]@half_zr
                        
            denomin = denomin + (kappa1 * kappa2) * mps_overlap(curr_mps1, curr_mps_shapes1,
                                                                curr_mps2, curr_mps_shapes2)
        

    overlap = tf.math.abs(nomin/tf.math.sqrt(denomin))
    overlap = 1-overlap#tf.cast(overlap, dtype=tf.float64)
    print(1-overlap)
    
    return overlap
    
    

def lcu_unitary_circuit_optimization(target_mps, kappas, lcu_mps): 

    half_id = tf.convert_to_tensor([[1., 0., 0., 0.],[0., 1., 0., 0.]], dtype=TFTYPE)
    half_zr = tf.reshape(tf.convert_to_tensor([1., 0.], dtype=TFTYPE), shape=(-1,1))
    
    kappas = np.array(kappas, dtype=NPTYPE)
    kappas = [kappa.tolist() for kappa in kappas]
    kappas = tf.Variable(kappas)

    target_isometry_list, target_shapes_list = quimb_mps_to_tf_mps(target_mps, canonical_form='left')
    target_isometry_list = list(map(tf.constant, target_isometry_list)) 
    target_isometry_list = list(map(lambda x: tf.cast(x, dtype=TFTYPE), target_isometry_list))
    
    lcu_isometry_list = []
    lcu_shapes_list = []
    for mps in lcu_mps:
        isometry_list, shapes_list = quimb_mps_to_tf_mps(mps, canonical_form='left')
        for shape, isometry in zip(shapes_list, isometry_list):
            
            if isometry.shape==(4,1): # eng 
                isometry = np.array([isometry, sp.linalg.null_space(isometry.T)[:,0].reshape(-1,1)]).squeeze().T
    
            elif isometry.shape==(2,2): # beginning
                temp = np.zeros((4,2),dtype=NPTYPE)
                temp[:2,:2] = isometry
                isometry = temp
            
    
            lcu_shapes_list.append(shape)
            lcu_isometry_list.append(isometry)
    
    
    no_of_layers, L = len(lcu_mps), target_mps.L
    
    tensor = np.zeros((no_of_layers*L, 4,2), dtype=NPTYPE)
    for it, isometry in enumerate(lcu_isometry_list):
        tensor[it,:,:] = isometry    
    tensor = tf.convert_to_tensor(tensor, dtype=TFTYPE)
    
    
    manifold = Manifold(4, 2, k=no_of_layers*L)
    euclidean_gradient = euclidean_hessian = None
    
    
    compute_overlap(tensor, lcu_shapes_list, 
                           target_isometry_list, target_shapes_list, 
                           kappas, 
                           no_of_layers, L,
                           half_id, half_zr)
    
    @pymanopt.function.tensorflow(manifold)
    def loss(x):
        return compute_overlap(x, lcu_shapes_list, 
                               target_isometry_list, target_shapes_list, 
                               kappas, 
                               no_of_layers, L,
                               half_id, half_zr)
    euclidean_gradient = euclidean_hessian = None
    

    problem = pymanopt.Problem(
        manifold,
        loss,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian)
    
    quiet = False
    
    # optimizer = TrustRegions(verbosity=2 * int(not quiet), max_time=7200)
    # estimated_spanning_set = optimizer.run( problem, Delta_bar=8 * np.sqrt(2), initial_point=tensor).point
    
    optimizer = ConjugateGradient(max_time=7200, max_iterations=3000)
    estimated_spanning_set = optimizer.run( problem, initial_point=tensor, ).point
    # optimizer = NelderMead()
    
        
    # for j in tqdm(range(iters)):        
    #     overlap = loss(tensor)
        
    #     # riemannian_optimizer.apply_gradients(zip(grad, lcu_isometry_list_var))    
        
    #     # learning_rate = riemannian_optimizer._get_hyper("learning_rate") * decay
    #     # riemannian_optimizer._set_hyper("learning_rate", learning_rate)
        
    #     # # vanilla optimization over kappas does not help
    #     # # optimizer.apply_gradients(zip([grad[-1]], [kappas]))        
    #     # overlaps_list.append(overlap)
    #     # grad_norm = np.linalg.norm(lcu_list_to_x(grad, shape_structure))
    #     # print(f' {overlap.numpy()=},\t{grad_norm=},\t')
        
    # print(overlap)
    # tf.print(f'final overlap (after {iters} iters): ', overlaps_list[-1])
    
    
if __name__ == "__main__":
    pass
    
    