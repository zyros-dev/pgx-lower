# Simplified string constant unit tests - no complex MLIR dependencies
add_executable(test_string_constants test_string_constants.cpp)

target_link_libraries(test_string_constants
    gtest 
    gtest_main
)

add_test(NAME StringConstantTest COMMAND test_string_constants)