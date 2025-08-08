# Util dialect unit tests

add_executable(test_util_basic test_util_basic.cpp)
target_link_libraries(test_util_basic
    gtest gtest_main
    MLIRUtilDialect
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
)
add_test(NAME UtilBasicTest COMMAND test_util_basic)