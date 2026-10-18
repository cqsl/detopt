# optimize 16 determinants for each of 4 states of N2 at a bond distance of 4.2 a.u.
sld-optimize-exc n2-4-2.xyz sto6g 10000 4 16 --nonorthogonal

# diagonalize in the full determinant space, compute the symmetry operators and the spin, point-group and Lz projectors
sld-postprocess-symm n2-4-2_sto6g_0_nstates0004_nper0016_step010000.h5 --nonorthogonal --project_spin 0 1 2

# diagonalize S^2 and Lz among degenerate states (or drop --auto for an interactive shell, see the lih example), writes _post_rotated.h5
sld-diagonalize-symm n2-4-2_sto6g_0_nstates0004_nper0016_step010000_post.h5 --auto

# assign irreps and print the states and projector weights
sld-show-symm n2-4-2_sto6g_0_nstates0004_nper0016_step010000_post_rotated.h5
