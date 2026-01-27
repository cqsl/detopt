# optimize 2 determinants for 1000 optimization steps, for the geometry in kr.xyz and in the ccpvdz basis
# here we use the (faster) non-svd version specified by --nonorthogonal, as typically we do not expect orthogonal determinants to appear (unless initialized manually)
sld-optimize kr.xyz ccpvdz 1000 2 --nonorthogonal
# compute <S^2> and variance
sld-postprocess kr_ccpvdz_0_ndet000002_step001000.h5

# now optimize again, but this time apply the particle-hole transformation to reduce the number of interacting particles
# store it under the name ./ph_....h5
sld-optimize kr.xyz ccpvdz 1000 2 --nonorthogonal --particle_hole --prefix ./ph_
sld-postprocess ph_kr_ccpvdz_0_ndet000002_step001000.h5
