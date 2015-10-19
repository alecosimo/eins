set(path "${PROJECT_SOURCE_DIR}/test")

M_ADD_TEST("${path}/fetiContextCreation.c"  "mpiexec -n 4 ./fetiContextCreation" "${EinsTests_libraries}")
M_ADD_TEST("${path}/fetiContextMatCreation.c"  "mpiexec -n 4 ./fetiContextMatCreation -p 1 -npx 2 -npy 2 -nex 4 -ney 4 -feti_fullyredundant" "${EinsTests_libraries}")
