# Unit tests for conversion passes
# Updated tests using current API

# Architectural compliance tests - validates proper LingoDB pipeline
add_executable(test_architectural_compliance test_architectural_compliance.cpp)

target_link_libraries(test_architectural_compliance
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRDBToDSAConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register architectural compliance tests with CTest
add_test(NAME ArchitecturalComplianceTest COMMAND test_architectural_compliance)

# Basic conversion tests
add_executable(test_conversion_basics test_conversion_basics.cpp)

target_link_libraries(test_conversion_basics
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRDBToDSAConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register basic conversion tests with CTest
add_test(NAME ConversionBasicsTest COMMAND test_conversion_basics)