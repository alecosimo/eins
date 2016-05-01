cmake_minimum_required(VERSION 2.8.8)

project(Eins)

# Path to cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/cmake-modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/cmakeSys")
include("${CMAKE_SOURCE_DIR}/cmake/cmakeSys/mainMacrosAndFunctions.cmake")

# Options for this project. Try to specify every option here
option(BUILD_DOCUMENTATION "Build the documentation"  ON)
option(ENABLE_PEDANTIC     "Enable pedantic warnings" ON)
option(FETI_DEBUG "Enable debug tests for FETI formulations" OFF)

# Declare and define every CMake variable in what follows
# ---------------------------------------------------
# Initialise and declare here every variable that you will use
set(extensionsCXX c cpp h hpp cxx)# list with ;separated values is recovered with "${extensionsCXX}"

# three targets files may be built: archive, library, and runtime.
# executables are runtime: right now does make too much sense
# static libraries archive
# module libraries library
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib CACHE PATH "Output directory for archives")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib CACHE PATH "Output directory for libraries")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin CACHE PATH "Output directory for executables")
set(CMAKE_DOC_OUTPUT_DIRECTORY     ${PROJECT_BINARY_DIR}/doc CACHE PATH "Output directory for documentation")
set(CMAKE_TEST_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/test CACHE PATH "Output directory for building and running tests")
set(CMAKE_CTEST_ARGS               "" CACHE STRING "Options for running CTest. Eg: -V")

# Configure a header file to pass some of the CMake settings to the source code
set(configure_file  "${PROJECT_BINARY_DIR}/einsConfig.h")
configure_file("${PROJECT_SOURCE_DIR}/einsConfig.h.in"  ${configure_file} )
include_directories("${PROJECT_BINARY_DIR}")

# --->>>> compiler configuration
# Determine which compiler is used
if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
  set(GCC_COMPILER 1)
  appendDefine2ConfFile(GCC_COMPILER 1)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
  set(INTEL_COMPILER 1)
  appendDefine2ConfFile(INTEL_COMPILER 1)
endif()

# Configure compiler flags
if(${GCC_COMPILER})
  include(commonCompilerFlagsGcc)#upto now only pedantic flags
endif()
# ---<<<< compiler configuration

# --->>>> Find sources for Eins by recursively including
# CMakeLists.txt and thus loading the sources for the target Eins
# (target: Eins; sources: Eins_sources). In this section we also find
# the libraries for linking Eins.

# Looking for sources by recursively including CMakelists.txt files
# from the root source "${PROJECT_SOURCE_DIR}/src"
M_INCLUDE_RECURSE("${PROJECT_SOURCE_DIR}/src" "*CMakeLists.txt")
include_directories("${PROJECT_SOURCE_DIR}/include")
M_INCLUDE_RECURSE("${PROJECT_SOURCE_DIR}/doc" "*CMakeLists.txt")

# Find requiered libraries and packages
find_library(MATH_LIB m)
find_package(MPI REQUIRED)
find_package(LAPACK REQUIRED)
find_package(PETSc REQUIRED)
include_directories(SYSTEM ${PETSC_INCLUDES})

# Set libraries for the Eins project, Eins_Libraries
set(Eins_libraries)
list(APPEND Eins_libraries ${PETSC_LIBRARIES} ${MATH_LIB} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY} ${LAPACK_LIBRARIES})

# Set libraries for the compiling the tests using the Eins library, EinsTests_Libraries
set(EINS_LIBRARY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libeins.so")
set(EinsTests_libraries)
list(APPEND EinsTests_libraries ${EINS_LIBRARY} ${PETSC_LIBRARIES} ${MATH_LIB} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY} ${LAPACK_LIBRARIES}) 

# Add tests. It must come after complete setting the libraries for linking
enable_testing()
string(REPLACE " " ";" option_list "${CMAKE_CTEST_ARGS}")
set(CMAKE_CTEST_COMMAND ctest ${option_list}) 
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
M_INCLUDE_RECURSE("${PROJECT_SOURCE_DIR}/test" "*CMakeLists.txt")

# add library to be built
add_library(libEins SHARED ${Eins_sources})
target_link_libraries(libEins ${Eins_libraries})
set_target_properties(libEins PROPERTIES OUTPUT_NAME "eins")

# add the install targets
# install(TARGETS Eins DESTINATION bin)
