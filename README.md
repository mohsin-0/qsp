# Quantum State Preparation

This repository provides an implementation of various methods for preparing tensor network states (specifically, 1D tensor network states) on a quantum computer. 

To use the package, you first need to specify a list of NumPy arrays that represent the MPS. You can then 
call different routines in the package to prepare the state.

## Installation
```
pip install git+https://github.com/mohsin-0/qsp.git@main
```

## Tutorial
[Usage tutorial](https://github.com/mohsin-0/qsp/blob/main/examples/state_prep_examples.ipynb) and some [benchmarks](https://github.com/mohsin-0/qsp/blob/main/examples/benchmarks.ipynb)


## Basic Example

```python
from qsp.tsp import MPSPreparation
import numpy as np
bond_dim, phys_dim = 4, 2

L=10
tensor_array = [np.random.rand(bond_dim,bond_dim,phys_dim) for _ in range(L)]
tensor_array[ 0] = np.random.rand(bond_dim,phys_dim)  # end points of mps
tensor_array[-1] = np.random.rand(bond_dim,phys_dim)
prep = MPSPreparation(tensor_array, shape='lrp')

prep.sequential_unitary_circuit(number_of_layers=4)
```

## References
1. [Encoding of matrix product states into quantum circuits of one-and two-qubit gates](), \ Shi-Ju Ran, Phys. Rev. A 101, 032310 \(2020\)\
2. [Variational power of quantum circuit tensor networks](), \ Reza Haghshenas,  Johnnie Gray, Andrew C Potter,  and Garnet Kin-Lic Chan, Phys. Rev. X 12, 011047 \(2022\)\
3. [Preentangling Quantum Algorithms--the Density Matrix Renormalization Group-assisted Quantum Canonical Transformation](), \ Mohsin Iqbal,  David Mu{\~n}oz Ramo and Henrik Dreyer, arXiv preprint arXiv:2209.07106 \(2022\)\
4. [Efficient adiabatic preparation of tensor network states](), Zhi-YuanWei, Daniel Malz and Ignacio J. Cirac, Phys. Rev. Research 5, L022037 \(2023\)\

