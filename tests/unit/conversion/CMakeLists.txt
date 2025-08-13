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