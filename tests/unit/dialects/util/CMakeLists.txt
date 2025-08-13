# Unit tests for Util dialect
add_executable(test_function_helper test_function_helper.cpp)
target_link_libraries(test_function_helper
    gtest gtest_main
    MLIRUtilDialect
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
)
add_test(NAME FunctionHelperTest COMMAND test_function_helper)