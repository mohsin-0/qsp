# Quantum State Preparation

This repository provides an implementation of various methods for preparing tensor network states (specifically, 1D tensor network states) on a quantum computer. 

To use the package, you first need to specify a list of NumPy arrays that represent the MPS. You can then 
call different routines in the package to prepare the state.

## Installation
```
pip install git+https://github.com/mohsin-0/qsp.git@main
```

## Tutorial
[Usage tutorial](https://github.com/mohsin-0/qsp/blob/main/examples/state_prep_examples.ipynb) and som [Benchmarks](https://github.com/mohsin-0/qsp/blob/main/examples/benchmarks.ipynb)


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
@article{ran2020encoding,
  title={Encoding of matrix product states into quantum circuits of one-and two-qubit gates},
  author={Ran, Shi-Ju},
  journal={Physical Review A},
  volume={101},
  number={3},
  pages={032310},
  year={2020},
  publisher={APS}
}

@article{haghshenas2022variational,
  title={Variational power of quantum circuit tensor networks},
  author={Haghshenas, Reza and Gray, Johnnie and Potter, Andrew C and Chan, Garnet Kin-Lic},
  journal={Physical Review X},
  volume={12},
  number={1},
  pages={011047},
  year={2022},
  publisher={APS}
}

@article{iqbal2022preentangling,
  title={Preentangling Quantum Algorithms--the Density Matrix Renormalization Group-assisted Quantum Canonical Transformation},
  author={Iqbal, Mohsin and Ramo, David Mu{\~n}oz and Dreyer, Henrik},
  journal={arXiv preprint arXiv:2209.07106},
  year={2022}
}

@article{wei2023efficient,
  title={Efficient adiabatic preparation of tensor network states},
  author={Wei, Zhi-Yuan and Malz, Daniel and Cirac, J Ignacio},
  journal={Physical Review Research},
  volume={5},
  number={2},
  pages={L022037},
  year={2023},
  publisher={APS}
}

```
