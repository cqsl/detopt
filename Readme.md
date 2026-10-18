This repository contains the code for the methods introduced in the papers
- _Precise Quantum Chemistry calculations with few Slater Determinants_ ([DOI:10.1038/s41467-026-70255-z](https://doi.org/10.1038/s41467-026-70255-z), [arXiv:2503.14502](https://arxiv.org/abs/2503.14502))
- _Variational low-energy subspaces for chemically accurate excited states_ ([arXiv:2606.05148](https://arxiv.org/abs/2606.05148))

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

ground state:
- `sld-optimize`: optimization, optionally of the spin and symmetry projected state
- `sld-optimize-partial`: optimization of the orbitals of only a subset of the determinants per step
- `sld-postprocess`: energy, variance and ⟨S²⟩

see [examples/ground_state](examples/ground_state)

excited states:
- `sld-optimize-exc`: optimization
- `sld-postprocess-symm`: matrix elements between the states, symmetry and spin projection, transition dipoles and 1-RDMs
- `sld-diagonalize-symm`: rotate (near) degenerate states into symmetry eigenstates
- `sld-show-symm`: print the states with their symmetry labels

see [examples/excited_state](examples/excited_state)

The output files are described in [docs/hdf5_file_format.md](docs/hdf5_file_format.md).
