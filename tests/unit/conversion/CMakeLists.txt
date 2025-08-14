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

# Test for RuntimeCallLowering in DBToStd conversion
add_executable(test_runtime_call_lowering test_runtime_call_lowering.cpp)
target_link_libraries(test_runtime_call_lowering
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
add_test(NAME RuntimeCallLoweringTest COMMAND test_runtime_call_lowering)

# Test for DB to Std parent module handling
add_executable(test_db_to_std_parent_module test_db_to_std_parent_module.cpp)
target_link_libraries(test_db_to_std_parent_module
    gtest 
    gtest_main
    MLIRDBToArrowStd
    MLIRDBDialect
    UtilDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    pgx_lower_logging
)
add_test(NAME DBToStdParentModuleTest COMMAND test_db_to_std_parent_module)

# Test for Constant Type Mapping fix (PostgreSQL OIDs)
add_executable(test_constant_type_mapping test_constant_type_mapping.cpp)
target_link_libraries(test_constant_type_mapping
    gtest 
    gtest_main
    MLIRDBToArrowStd
    MLIRDBDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass
    MLIRIR
    pgx_lower_logging
)
add_test(NAME ConstantTypeMappingTest COMMAND test_constant_type_mapping)