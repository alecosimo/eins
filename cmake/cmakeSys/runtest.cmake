# solution based on http://cmake.3232098.n2.nabble.com/testing-with-standard-output-td3256689.html
separate_arguments(test_cmd)

execute_process(
   COMMAND ${test_cmd} OUTPUT_FILE "${output_test}"
   WORKING_DIRECTORY ${working_dir}
)

execute_process(
   COMMAND ${CMAKE_COMMAND} -E compare_files ${output_blessed} ${output_test}
   RESULT_VARIABLE test_not_successful
   WORKING_DIRECTORY ${working_dir}
)

if(test_not_successful)
   message(SEND_ERROR "${output_test} does not match ${output_blessed}!")
endif( test_not_successful ) 