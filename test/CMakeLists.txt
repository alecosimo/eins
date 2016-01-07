set(path "${PROJECT_SOURCE_DIR}/test")

M_ADD_TEST("${path}/Oscillator2.c"  "mpiexec -n 4 ./Oscillator2 -ts_max_steps 10 -ts_alpha_adapt -damping 0.1" "${EinsTests_libraries}")

M_ADD_TEST("${path}/vecUnasmHDF5.c"  "mpiexec -n 3 ./vecUnasmHDF5 -log_destruction" "${EinsTests_libraries}")

M_ADD_TEST_CMP("${path}/testVecUAsm.c"  "mpiexec -n 3 ./testVecUAsm -log_destruction" "${EinsTests_libraries}" "res_testVecUAsm.txt")
M_ADD_TEST_CMP("${path}/fetiContextCreation.c"  "mpiexec -n 4 ./fetiContextCreation -log_destruction" "${EinsTests_libraries}" "res_fetiContextCreation.txt")

#M_ADD_TEST_CMP("${path}/ex7.c"  "mpiexec -n 2 ./ex7 -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -ksp_type pjgmres -log_destruction" "${EinsTests_libraries}" "res_ex7.txt")

M_ADD_BINARY_TEST("${path}/poissonFD.c" "${EinsTests_libraries}")
M_ADD_COMMAND_TEST_CMP("${path}/poissonFD.c" "1" "mpiexec -n 4 ./poissonFD -npx 2 -npy 2 -nex 4 -ney 4 -feti_interface_ksp_monitor_short -log_destruction" "res_poissonFD1.txt")

M_ADD_BINARY_TEST("${path}/fetiContextMatCreation.c" "${EinsTests_libraries}")
M_ADD_COMMAND_TEST_CMP("${path}/fetiContextMatCreation.c" "1" "mpiexec -n 4 ./fetiContextMatCreation -p 1 -npx 4 -npy 1 -nex 8 -ney 2 -feti_interface_ksp_monitor_short -log_destruction" "res_fetiContextMatCreation1.txt")
M_ADD_COMMAND_TEST_CMP("${path}/fetiContextMatCreation.c" "2" "mpiexec -n 4 ./fetiContextMatCreation -p 1 -npx 2 -npy 2 -nex 4 -ney 4 -feti_pc_dirichlet_pc_factor_mat_solver_package mumps -feti_pc_dirichlet_mat_mumps_icntl_7 2 -feti_fullyredundant -feti_scaling_type scmultiplicity -feti_interface_ksp_monitor_short -log_destruction" "res_fetiContextMatCreation2.txt")# -feti_interface_ksp_gmres_cgs_refinement_type refine_always


#mpiexec -n 4 ./fetiContextMatCreation -p 1 -npx 2 -npy 2 -nex 4 -ney 4 -feti_interface_ksp_monitor_short -feti_fullyredundant 0
#mpiexec -n 4 ./fetiContextMatCreation -p 1 -npx 4 -npy 1 -nex 8 -ney 2 -feti_interface_ksp_monitor_short