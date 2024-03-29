# Add a test
# filePath: file name with extension and complete path
# command: command to run the test: it must be a STRING
macro(M_ADD_TEST filePath command libraries)
  get_filename_component(target ${filePath} NAME_WE)
  add_executable("test_${target}" EXCLUDE_FROM_ALL ${filePath})
  set_property(TARGET "test_${target}" PROPERTY OUTPUT_NAME ${target})
  set_property(TARGET "test_${target}" PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_TEST_OUTPUT_DIRECTORY})
  target_link_libraries("test_${target}" ${libraries})
  string(REPLACE " " ";" command_list ${command})
  add_test(NAME "test_${target}" COMMAND ${command_list} WORKING_DIRECTORY ${CMAKE_TEST_OUTPUT_DIRECTORY})
  add_dependencies(check "test_${target}")  
endmacro()

# Add a test and compare results
# filePath: file name with extension and complete path
# command: command to run the test: it must be a STRING
# file_cmp: file for comparing output
macro(M_ADD_TEST_CMP filePath command libraries file_cmp)
  get_filename_component(target ${filePath} NAME_WE)
  add_executable("test_${target}" EXCLUDE_FROM_ALL ${filePath})
  set_property(TARGET "test_${target}" PROPERTY OUTPUT_NAME ${target})
  set_property(TARGET "test_${target}" PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_TEST_OUTPUT_DIRECTORY})
  target_link_libraries("test_${target}" ${libraries})
  add_test("test_${target}"
    ${CMAKE_COMMAND}
    -D test_cmd=${command}
    -D output_blessed=${CMAKE_SOURCE_DIR}/test/results/${file_cmp}
    -D output_test=${CMAKE_TEST_OUTPUT_DIRECTORY}/res_test_${target}_${id}
    -D working_dir=${CMAKE_TEST_OUTPUT_DIRECTORY}
    -P ${CMAKE_SOURCE_DIR}/cmake/cmakeSys/runtest.cmake
    )
  add_dependencies(check "test_${target}")  
endmacro()


# Add a test to be compiled
# filePath: file name with extension and complete path
macro(M_ADD_BINARY_TEST filePath libraries)
  get_filename_component(target ${filePath} NAME_WE)
  add_executable("test_${target}" EXCLUDE_FROM_ALL ${filePath})
  set_property(TARGET "test_${target}" PROPERTY OUTPUT_NAME ${target})
  set_property(TARGET "test_${target}" PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_TEST_OUTPUT_DIRECTORY})
  target_link_libraries("test_${target}" ${libraries})
  add_dependencies(check "test_${target}")  
endmacro()


# Add a command to already existing binary test and compare results
# filePath: file name with extension and complete path
# id: id for the test
# command: command to run the test: it must be a STRING
# file_cmp: file for comparing output
macro(M_ADD_COMMAND_TEST_CMP filePath id command file_cmp)
  get_filename_component(target ${filePath} NAME_WE)
  add_test("test_${target}_${id}"
    ${CMAKE_COMMAND}
    -D test_cmd=${command}
    -D output_blessed=${CMAKE_SOURCE_DIR}/test/results/${file_cmp}
    -D output_test=${CMAKE_TEST_OUTPUT_DIRECTORY}/res_test_${target}_${id}
    -D working_dir=${CMAKE_TEST_OUTPUT_DIRECTORY}
    -P ${CMAKE_SOURCE_DIR}/cmake/cmakeSys/runtest.cmake
    )
endmacro()


# Add a command to already existing binary test
# filePath: file name with extension and complete path
# id: id for the test
# command: command to run the test: it must be a STRING
macro(M_ADD_COMMAND_TEST filePath id command)
  get_filename_component(target ${filePath} NAME_WE)
  string(REPLACE " " ";" command_list ${command})
  add_test(NAME "test_${target}_${id}" COMMAND ${command_list} WORKING_DIRECTORY ${CMAKE_TEST_OUTPUT_DIRECTORY})
endmacro()


# Append define to configureFile "EinsConfig.h" (identified by the
# variable configure_file)
# var: name of the variable
# value: value assigned to the variable
function(appendDefine2ConfFile var value)
  file(APPEND ${configure_file} "#define ${var} ${value}")
endfunction()


# add sources to a given target, eg to the Vicent binary
# extensions: eg c cpp h---> variables extensionsCXX, extensionsSWIG,etc.
# path: the path where to look for sources with the previous extensions
#
# It ALSO includes the directories
macro(M_SOURCES_FOR_TARGET target extensions path)
  set(${target}_sources ${${target}_sources})
  foreach(e ${extensions})
    file(GLOB files "${path}/*.${e}")

    list(APPEND ${target}_sources ${files})
  endforeach()
  include_directories(${path})
endmacro()


#Search recursively files from root with regular expression "regexp" and include them to th project
#IN: *regexp: regular expression
#    *root: root path for search recursively
macro(M_INCLUDE_RECURSE root regexp)
  file(GLOB_RECURSE files "${root}/${regexp}")
  foreach(e ${files})
    include(${e})
  endforeach()
endmacro()

