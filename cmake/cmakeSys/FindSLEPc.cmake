# Try to find SLEPC
#
#  SLEPC_FOUND        - system has SLEPC
#  SLEPC_INCLUDES     - the SLEPC include directories
#  SLEPC_LIBRARIES    - Link these to use SLEPC


find_path(SLEPC_DIR include/slec.h HINTS ENV SLEPC_DIR DOC "SLEPC top-level directory")
set(SLEPC_ARCH $ENV{PETSC_ARCH} CACHE STRING "SLEPC build")
find_path(SLEPC_INCLUDE_DIR slepc.h HINTS "${SLEPC_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH DOC "SLEPC include path")
mark_as_advanced(SLEPC_INCLUDE_DIR)
set(SLEPC_INCLUDES ${SLEPC_INCLUDE_DIR} CACHE PATH "SLEPC include paths" FORCE)
find_library(SLEPC_LIBRARIES NAMES slepc HINTS "${SLEPC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/lib" "lib" NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEPC
  "SLEPC could not be found.  Be sure to set SLEPC_DIR and PETSC_ARCH."
  SLEPC_INCLUDES SLEPC_LIBRARIES)
