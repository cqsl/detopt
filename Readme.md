This repository contains the code for the method introduced in the paper [_Precise Quantum Chemistry calculations with few Slater Determinants_](https://arxiv.org/abs/2503.14502)

It is a python library using jax, and is generally intended to be run on gpu, or cpu for small simulations.
An experimental parallelized implementation is available (keeping the serial cholesky decomposition) 
and automatically used when multiple jax devices are present.

### to install:

```bash
git clone https://github.com/cqsl/detopt.git
cd detopt
pip install .
```

### to run: 
see examples




