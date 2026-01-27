# optimize 8 determinants for 5000 optimization steps, for the geometry in o2.xyz and in the sto-6g basis
# this should converge to the triplet state
# here we use the (faster) non-svd version specified by --nonorthogonal
sld-optimize o2.xyz sto6g 5000 8 --nonorthogonal
# compute <S^2> and variance
echo M=0, S=1
sld-postprocess o2_sto6g_0_ndet000008_step005000.h5

# now optimize to the tripet in the M=1 sector
# need to use dense eigendecompositon code path for the solver,
# because the nullspace projection has not been implemented yet for this case
sld-optimize o2.xyz sto6g 5000 8 --nonorthogonal --spin 2 --dense_solver
echo M=1, S=1
sld-postprocess o2_sto6g_2_ndet000008_step005000.h5

# now optimize to the singlet state by adding a penalty term
sld-optimize o2.xyz sto6g 5000 8 --nonorthogonal --s2_loss 0.1 --prefix loss01_
echo M=0, S=0
sld-postprocess loss01_o2_sto6g_0_ndet000008_step005000.h5
