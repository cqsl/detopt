# optimize 8 determinants for each of 5 states of LiH in the sto-6g basis
sld-optimize-exc lih.xyz sto6g 5000 5 8 --nonorthogonal

# diagonalize in the full determinant space, compute the symmetry operators and the spin, point-group and Lz projectors
sld-postprocess-symm lih_sto6g_0_nstates0005_nper0008_step005000.h5 --nonorthogonal --project_spin 0 1 2

# diagonalize S^2 and Lz among degenerate states (or drop --auto for an interactive shell), writes _post_rotated.h5
sld-diagonalize-symm lih_sto6g_0_nstates0005_nper0008_step005000_post.h5 --auto

# or interactively, e.g. for the 3Π pair k = 3, 4 of the table:
#   sld-diagonalize-symm lih_sto6g_0_nstates0005_nper0008_step005000_post.h5
#   diagonalize([3, 4], 'lz')
#   save()

# assign irreps and print the states and projector weights
sld-show-symm lih_sto6g_0_nstates0005_nper0008_step005000_post_rotated.h5
