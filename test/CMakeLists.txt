set(path "${PROJECT_SOURCE_DIR}/test")

M_ADD_TEST("${path}/matMatBeamTiming.c" "mpiexec -n 4 ./matMatBeamTiming")
M_ADD_TEST("${path}/matWithoutError.c"  "mpiexec -n 4 ./matWithoutError")
