# Add a test
# filePath: file name with extension and complete path
# command: command to run the test: it must be a STRING
macro(M_ADD_TEST filePath command)
  get_filename_component(target ${filePath} NAME_WE)
  add_executable("test_${target}" EXCLUDE_FROM_ALL ${filePath})
  set_property(TARGET "test_${target}" PROPERTY OUTPUT_NAME ${target})
  set_property(TARGET "test_${target}" PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_TEST_OUTPUT_DIRECTORY})
  target_link_libraries("test_${target}" ${EinsTests_libraries})
  string(REPLACE " " ";" command_list ${command})
  add_test(NAME "test_${target}" COMMAND ${command_list} WORKING_DIRECTORY ${CMAKE_TEST_OUTPUT_DIRECTORY})
  add_dependencies(check "test_${target}")  
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

