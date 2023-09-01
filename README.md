# Quantum State Preparation

This repository contains an implemetation of different methods to prepapre tensor network states (mostly in 1d) on a quantum computer.

As an input ones needs to specify list of numpy arrays specifying MPS and then call different routines to use the package.

## Installation
```
pip install git+https://github.com/mohsin-0/qsp.git@main
```

## Tutorial
[Usage tutorial](https://github.com/mohsin-0/qsp/blob/main/examples/state_prep_examples.ipynb)


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

```
@article{iqbal2022preentangling,
  title={Preentangling Quantum Algorithms--the Density Matrix Renormalization Group-assisted Quantum Canonical Transformation},
  author={Iqbal, Mohsin and Ramo, David Mu{\~n}oz and Dreyer, Henrik},
  journal={arXiv preprint arXiv:2209.07106},
  year={2022}
}
```
