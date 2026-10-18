# optimize 4 determinants for each of 3 states of water in the sto-6g basis
sld-optimize-exc water.xyz sto6g 1000 3 4 --nonorthogonal

# diagonalize in the full determinant space, compute the symmetry operators and the singlet and triplet projectors
sld-postprocess-symm water_sto6g_0_nstates0003_nper0004_step001000.h5 --nonorthogonal --project_spin 0 1

# diagonalize S^2 and the point group among near-degenerate states (or drop --auto for an interactive shell, see the lih example), writes _post_rotated.h5
sld-diagonalize-symm water_sto6g_0_nstates0003_nper0004_step001000_post.h5 --auto

# assign irreps and print the states and projector weights
sld-show-symm water_sto6g_0_nstates0003_nper0004_step001000_post_rotated.h5
