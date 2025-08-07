# Unit tests for conversion passes
# Based on dialect test patterns

add_executable(test_relalg_to_dsa test_relalg_to_dsa.cpp)

target_link_libraries(test_relalg_to_dsa
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDSADialect
    MLIRRelAlgToDSAConversion
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register the test with CTest
add_test(NAME RelAlgToDSAConversionTest COMMAND test_relalg_to_dsa)

# Comprehensive lowering tests
add_executable(test_relalg_to_dsa_lowering_comprehensive test_relalg_to_dsa_lowering_comprehensive.cpp)

target_link_libraries(test_relalg_to_dsa_lowering_comprehensive
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDSADialect
    MLIRRelAlgToDSAConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register comprehensive lowering tests with CTest
add_test(NAME RelAlgToDSALoweringComprehensiveTest COMMAND test_relalg_to_dsa_lowering_comprehensive)

# Working lowering tests (focused on successful transformations)
add_executable(test_relalg_to_dsa_working_lowering test_relalg_to_dsa_working_lowering.cpp)

target_link_libraries(test_relalg_to_dsa_working_lowering
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDSADialect
    MLIRRelAlgToDSAConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register working lowering tests with CTest
add_test(NAME WorkingLoweringTest COMMAND test_relalg_to_dsa_working_lowering)

# Focused lowering validation tests
add_executable(test_focused_lowering_validation test_focused_lowering_validation.cpp)

target_link_libraries(test_focused_lowering_validation
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDSADialect
    MLIRRelAlgToDSAConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register focused validation tests with CTest
add_test(NAME FocusedLoweringValidationTest COMMAND test_focused_lowering_validation)

# MaterializeOp lowering validation tests
add_executable(test_materialize_op_lowering test_materialize_op_lowering.cpp)

target_link_libraries(test_materialize_op_lowering
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDSADialect
    MLIRRelAlgToDSAConversion
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register MaterializeOp lowering tests with CTest
add_test(NAME MaterializeOpLoweringTest COMMAND test_materialize_op_lowering)

# Simple MaterializeOp lowering validation tests
add_executable(test_materialize_op_simple test_materialize_op_simple.cpp)

target_link_libraries(test_materialize_op_simple
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDSADialect
    MLIRRelAlgToDSAConversion
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register simple MaterializeOp tests with CTest
add_test(NAME SimpleMaterializeOpLoweringTest COMMAND test_materialize_op_simple)