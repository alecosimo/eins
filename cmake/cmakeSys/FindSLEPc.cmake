# Try to find SLEPC
#
#  SLEPC_FOUND        - system has SLEPC
#  SLEPC_INCLUDES     - the SLEPC include directories
#  SLEPC_LIBRARIES    - Link these to use SLEPC


find_path(SLEPC_DIR include/slepc.h HINTS ENV SLEPC_DIR DOC "SLEPC top-level directory")

if (NOT(${SLEPC_DIR} STREQUAL "SLEPC_DIR-NOTFOUND"))
  set(SLEPC_ARCH $ENV{PETSC_ARCH} CACHE STRING "SLEPC build")
  find_path (SLEPC_INCLUDE_CONF slepcconf.h HINTS "${SLEPC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/include" "bmake/${PETSC_ARCH}" NO_DEFAULT_PATH)
  find_path(SLEPC_INCLUDE_DIR slepc.h HINTS "${SLEPC_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH DOC "SLEPC include path")
  mark_as_advanced(SLEPC_INCLUDE_DIR SLEPC_INCLUDE_CONF)
  set(SLEPC_INCLUDES ${SLEPC_INCLUDE_DIR} ${SLEPC_INCLUDE_CONF} CACHE PATH "SLEPC include paths" FORCE)
  mark_as_advanced(SLEPC_INCLUDES)
  find_library(SLEPC_LIBRARIES NAMES slepc HINTS "${SLEPC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/lib" "lib" NO_DEFAULT_PATH)
  mark_as_advanced(SLEPC_LIBRARIES)
endif (NOT(${SLEPC_DIR} STREQUAL "SLEPC_DIR-NOTFOUND"))

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEPC
  "SLEPC could not be found.  Be sure to set SLEPC_DIR and PETSC_ARCH."
  SLEPC_INCLUDES SLEPC_LIBRARIES)
