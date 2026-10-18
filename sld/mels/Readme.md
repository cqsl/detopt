Library to compute matrix elements between non-orthogonal slater determinants, with two implementations:
  - pairwise (`_src/pairwise`): computes each pair of determinants separately
  - hybrid (`_src/hybrid`): contracts the Hamiltonian with the orbitals of each bra once and reuses it for all kets; typically faster than pairwise, and the default for UHF

Both have a "stable" version (SVD, any determinants) and a faster "unstable" one (LU, non-orthogonal determinants only).

Features:
  - support for GHF-type and UHF-type determinants
  - overlap, matrix elements of H, and matrix elements of H^2 (not in the unstable pairwise version)
  - Gradients and Hessians of the matrix elements (except H^2)
