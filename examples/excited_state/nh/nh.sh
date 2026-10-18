# optimize 8 determinants for each of 4 states of NH, open shell with Sz = 1 (--spin 2), in the sto-6g basis
sld-optimize-exc nh.xyz sto6g 5000 4 8 --spin 2 --nonorthogonal

# diagonalize in the full determinant space, compute the symmetry operators and the spin, point-group and Lz projectors (S >= Sz)
sld-postprocess-symm nh_sto6g_2_nstates0004_nper0008_step005000.h5 --nonorthogonal --project_spin 1 2

# diagonalize S^2 and Lz among degenerate states (or drop --auto for an interactive shell), writes _post_rotated.h5
sld-diagonalize-symm nh_sto6g_2_nstates0004_nper0008_step005000_post.h5 --auto

# or interactively, e.g. for the 3Π pair k = 1, 2 of the table:
#   sld-diagonalize-symm nh_sto6g_2_nstates0004_nper0008_step005000_post.h5
#   diagonalize([1, 2], 'lz')
#   save()

# assign irreps and print the states and projector weights
sld-show-symm nh_sto6g_2_nstates0004_nper0008_step005000_post_rotated.h5

# GHF determinants: Sz is not fixed, so the 3Σ- ground state comes as three states, then the 1Δ pair
sld-optimize-exc nh.xyz sto6g 1000 5 8 --spin 2 --ghf --prefix ghf_
# the GHF spin projector sums over all Sz
sld-postprocess-symm ghf_nh_sto6g_2_nstates0005_nper0008_step001000.h5 --project_spin 0 1 2
# rotates the 3Σ- states to Sz = -1, 0, 1 and the 1Δ pair to Lz = ±2
sld-diagonalize-symm ghf_nh_sto6g_2_nstates0005_nper0008_step001000_post.h5 --auto
sld-show-symm ghf_nh_sto6g_2_nstates0005_nper0008_step001000_post_rotated.h5

# or by hand, with the k of the table:
#   sld-diagonalize-symm ghf_nh_sto6g_2_nstates0005_nper0008_step001000_post.h5
#   diagonalize([0, 1, 2], 'sz')
#   diagonalize([3, 4], 'lz')
#   save()
