# Unit tests for conversion passes

# Test for StandardToLLVM unified conversion pass
add_executable(test_standard_to_llvm test_standard_to_llvm.cpp)
target_link_libraries(test_standard_to_llvm
    gtest 
    gtest_main
    StandardToLLVM
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLLVMDialect
    UtilDialect
    MLIRPass
    MLIRIR
    pgx_lower_logging
)
add_test(NAME StandardToLLVMTest COMMAND test_standard_to_llvm)

# Test for DBToStd conversion pass
add_executable(test_db_to_std test_db_to_std.cpp)
target_link_libraries(test_db_to_std
    gtest 
    gtest_main
    MLIRDBToArrowStd
    MLIRDBDialect
    MLIRDSADialect
    UtilDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass
    MLIRIR
    pgx_lower_logging
)
add_test(NAME DBToStdTest COMMAND test_db_to_std)

# Test for RelAlgToDB conversion pass
add_executable(test_relalg_to_db test_relalg_to_db.cpp)
target_link_libraries(test_relalg_to_db
    gtest 
    gtest_main
    MLIRRelAlgToDBPasses
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    UtilDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass
    MLIRIR
    pgx_lower_logging
)
add_test(NAME RelAlgToDBTest COMMAND test_relalg_to_db)