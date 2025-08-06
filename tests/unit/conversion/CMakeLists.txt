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