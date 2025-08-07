# Unit tests for conversion passes
# CRITICAL ARCHITECTURAL FIX: RelAlgToDSA bypass tests REMOVED
# All tests now follow proper LingoDB pipeline: RelAlg → DB → DSA

# NEW ARCHITECTURAL TESTS - Following LingoDB pipeline

# RelAlg to DB conversion tests
add_executable(test_relalg_to_db test_relalg_to_db.cpp)

target_link_libraries(test_relalg_to_db
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRRelAlgToDBConversion
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

# Register RelAlg to DB tests with CTest
add_test(NAME RelAlgToDBConversionTest COMMAND test_relalg_to_db)

# DB to DSA conversion tests
add_executable(test_db_to_dsa test_db_to_dsa.cpp)

target_link_libraries(test_db_to_dsa
    gtest 
    gtest_main
    MLIRDBDialect
    MLIRDSADialect
    MLIRDBToDSAConversion
    MLIRRelAlgDialect  # For end-to-end testing
    MLIRRelAlgToDBConversion  # For end-to-end testing
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

# Register DB to DSA tests with CTest
add_test(NAME DBToDSAConversionTest COMMAND test_db_to_dsa)

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