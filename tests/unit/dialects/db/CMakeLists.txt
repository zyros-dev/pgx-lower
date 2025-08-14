# DB dialect unit tests
add_executable(test_db_to_std_pass test_db_to_std_pass.cpp)
target_link_libraries(test_db_to_std_pass
    gtest gtest_main
    MLIRDBDialect
    MLIRDSADialect
    MLIRUtilDialect
    MLIRDBToArrowStd
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFunc
    MLIRArith
)
add_test(NAME DBToStdPassTest COMMAND test_db_to_std_pass)