set(path "${PROJECT_SOURCE_DIR}/test")

# mpiexec -n 2 ./poissonFD -npx 2 -npy 1 -nex 4 -ney 2 -feti_interface_ksp_monitor_short -feti_interface_pc_type pcfeti_dirichlet -feti_scaling_type scmultiplicity

# mpiexec -n 4 ./poissonFD -npx 2 -npy 2 -nex 4 -ney 4 -feti_interface_ksp_monitor_short -feti_interface_pc_type pcfeti_dirichlet -feti_scaling_type scmultiplicity

M_ADD_BINARY_TEST("${path}/poissonFDkspfetiTimes.c" "${EinsTests_libraries}")
M_ADD_BINARY_TEST("${path}/poissonFD.c" "${EinsTests_libraries}")
M_ADD_COMMAND_TEST_CMP("${path}/poissonFD.c" "1" "mpiexec -n 4 ./poissonFD -npx 2 -npy 2 -nex 4 -ney 4 -feti_interface_ksp_monitor_short -log_destruction" "res_poissonFD1.txt")

M_ADD_BINARY_TEST("${path}/poissonFDkspfeti.c" "${EinsTests_libraries}")
M_ADD_COMMAND_TEST_CMP("${path}/poissonFDkspfeti.c" "1" "mpiexec -n 4 ./poissonFDkspfeti -npx 2 -npy 2 -nex 4 -ney 4 -feti_interface_ksp_monitor_short -log_destruction" "res_poissonFDkspfeti1.txt")
M_ADD_COMMAND_TEST_CMP("${path}/poissonFDkspfeti.c" "2" "mpiexec -n 4 ./poissonFDkspfeti -npx 4 -npy 1 -nex 8 -ney 2 -feti_interface_ksp_monitor_short -log_destruction" "res_poissonFDkspfeti2.txt")

if (PETSC_VERSION VERSION_GREATER 3.6.4)
  M_ADD_TEST("${path}/Oscillator2.c"  "mpiexec -n 4 ./Oscillator2 -ts_max_steps 10 -ts_alpha_adapt -damping 0.1" "${EinsTests_libraries}")
endif (PETSC_VERSION VERSION_GREATER 3.6.4)

M_ADD_TEST("${path}/vecUnasmHDF5.c"  "mpiexec -n 3 ./vecUnasmHDF5 -log_destruction" "${EinsTests_libraries}")

M_ADD_TEST_CMP("${path}/testVecUAsm.c"  "mpiexec -n 3 ./testVecUAsm -log_destruction" "${EinsTests_libraries}" "res_testVecUAsm.txt")
M_ADD_TEST_CMP("${path}/fetiContextCreation.c"  "mpiexec -n 4 ./fetiContextCreation -log_destruction" "${EinsTests_libraries}" "res_fetiContextCreation.txt")


