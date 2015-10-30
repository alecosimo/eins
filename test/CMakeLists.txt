set(path "${PROJECT_SOURCE_DIR}/test")

M_ADD_TEST("${path}/fetiContextCreation.c"  "mpiexec -n 4 ./fetiContextCreation" "${EinsTests_libraries}")
M_ADD_TEST("${path}/ex7.c"  "mpiexec -n 2 ./ex7 -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -ksp_type pjgmres" "${EinsTests_libraries}")#
M_ADD_BINARY_TEST("${path}/fetiContextMatCreation.c" "${EinsTests_libraries}")
M_ADD_COMMAND_TEST("${path}/fetiContextMatCreation.c" "1" "mpiexec -n 4 ./fetiContextMatCreation -p 1 -npx 4 -npy 1 -nex 8 -ney 2")
M_ADD_COMMAND_TEST("${path}/fetiContextMatCreation.c" "2" "mpiexec -n 4 ./fetiContextMatCreation -p 1 -npx 2 -npy 2 -nex 4 -ney 4 -feti_pc_dirichlet_pc_factor_mat_solver_package mumps -feti_pc_dirichlet_mat_mumps_icntl_7 2 -feti_fullyredundant -feti_scaling_type scmultiplicity")#-feti_interface_ksp_monitor_short -feti_interface_ksp_gmres_cgs_refinement_type refine_always