# N2 at 1.5 times the equilibrium distance in sto-6g, where the unprojected UHF determinants break the spin symmetry

# 8 determinants without projection
sld-optimize n2_stretched.xyz sto6g 1000 8 --nonorthogonal --seed 1

# projection after variation (PAV): ⟨Ĥ P̂⟩ and ⟨P̂⟩ for S = 0, the point group and Lz
sld-postprocess-symm n2_stretched_sto6g_0_ndet000008_step001000.h5 --nonorthogonal --project_spin 0 --projected_energy
# the spin and irrep weights of the state and the projected energy of each sector
sld-show-symm n2_stretched_sto6g_0_ndet000008_step001000_post.h5

# variation after projection (VAP): the optimizer minimizes ⟨Ĥ P̂⟩/⟨P̂⟩, the cost per step grows with the number of grid points
# the singlet (2 grid points)
sld-optimize n2_stretched.xyz sto6g 1000 8 --nonorthogonal --seed 1 --project_spin 0 --prefix vap_s0_
# the singlet and A1g of the subgroup D4h (2 x 16 grid points); sld-optimize prints the available irreps
sld-optimize n2_stretched.xyz sto6g 1000 8 --nonorthogonal --seed 1 --project_spin 0 --project_irrep A1g --linear_subgroup d4h --prefix vap_d4h_
# the full D∞h projector onto 1Σg+ is exact but expensive (2 x 4 x 29 grid points)
sld-optimize n2_stretched.xyz sto6g 200 8 --nonorthogonal --seed 1 --project_spin 0 --project_irrep g+ --project_lz 0 --prefix vap_sigma_
