# optimize 2 determinants for 1000 optimization steps, for the geometry in n2.xyz and in the sto-6g basis
# here we use the (faster) non-svd version specified by --nonorthogonal
sld-optimize n2.xyz sto6g 1000 2 --nonorthogonal
# compute <S^2> and variance
sld-postprocess n2_sto6g_0_ndet000002_step001000.h5

# print energy, variance and <S^2>
h5dump -g postprocess n2_sto6g_0_ndet000002_step001000_var.h5

# now optimzie 4 determinants, starting from the 2 pre-optimized ones above, adding two randomly initialized ones
sld-optimize n2.xyz sto6g 1000 4 --nonorthogonal --start n2_sto6g_0_ndet000002_step001000.h5

# finally optimize a subset of 4 determinants at every step for with total number of 32 determinants
sld-optimize-partial n2.xyz sto6g 1000 32 --nonorthogonal --partialopt 4 --prefix partial_
