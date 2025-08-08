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

# Phase 3a integration tests
add_executable(test_phase3a_integration test_phase3a_integration.cpp)

target_link_libraries(test_phase3a_integration
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
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

# Register Phase 3a integration tests with CTest
add_test(NAME Phase3AIntegrationTest COMMAND test_phase3a_integration)

# Functional conversion tests - tests that actually run conversions and verify output
add_executable(test_functional_conversions test_functional_conversions.cpp)

target_link_libraries(test_functional_conversions
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
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

# Register functional conversion tests with CTest
add_test(NAME FunctionalConversionsTest COMMAND test_functional_conversions)

# Type creation tests - validates basic type functionality
add_executable(test_type_creation test_type_creation.cpp)

target_link_libraries(test_type_creation
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRFuncDialect
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register type creation tests with CTest
add_test(NAME TypeCreationTest COMMAND test_type_creation)

# DSA to LLVM conversion tests - Phase 4a implementation
add_executable(test_dsa_to_llvm test_dsa_to_llvm.cpp)

target_link_libraries(test_dsa_to_llvm
    gtest 
    gtest_main
    MLIRDSADialect
    MLIRLLVMDialect
    MLIRDSAToLLVMConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTransformUtils
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register DSA to LLVM tests with CTest
add_test(NAME DSAToLLVMTest COMMAND test_dsa_to_llvm)

# RelAlg to DB conversion tests - Phase 4c-2 implementation (with DSA)
add_executable(test_relalg_to_db test_relalg_to_db.cpp)

target_link_libraries(test_relalg_to_db
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register RelAlg to DB tests with CTest
add_test(NAME RelAlgToDBTest COMMAND test_relalg_to_db)

# MaterializeOp Phase 4c-2 tests - Mixed DB+DSA operations
add_executable(test_materialize_phase4c2 test_materialize_phase4c2.cpp)

target_link_libraries(test_materialize_phase4c2
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register MaterializeOp Phase 4c-2 tests with CTest
add_test(NAME MaterializePhase4c2Test COMMAND test_materialize_phase4c2)

# Translator infrastructure tests
add_executable(test_translator_infrastructure test_translator_infrastructure_simple.cpp)

target_link_libraries(test_translator_infrastructure
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
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

add_test(NAME TranslatorInfrastructureTest COMMAND test_translator_infrastructure)

# BaseTable streaming tests - Phase 4c-2 implementation
add_executable(test_basetable_streaming test_basetable_streaming.cpp)

target_link_libraries(test_basetable_streaming
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
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

# Register BaseTable streaming tests with CTest
add_test(NAME BaseTableStreamingTest COMMAND test_basetable_streaming)

# Streaming memory validation tests - Phase 4c-2 memory efficiency
add_executable(test_streaming_memory_validation test_streaming_memory_validation.cpp)

target_link_libraries(test_streaming_memory_validation
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
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

# Register streaming memory validation tests with CTest
add_test(NAME StreamingMemoryValidationTest COMMAND test_streaming_memory_validation)

# MaterializeOp DB operations tests - Corrected architecture (DB ops only)
add_executable(test_materialize_db_ops test_materialize_db_ops.cpp)

target_link_libraries(test_materialize_db_ops
    gtest 
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRPass
    MLIRTransforms
    MLIRDialectConversion
    MLIRIR
    MLIRSupport
    LLVMSupport
    pgx_logging
)

# Register MaterializeOp DB operations tests with CTest
add_test(NAME MaterializeDBOpsTest COMMAND test_materialize_db_ops)

# Phase 4c-4: Complete Pipeline Integration tests
add_executable(test_complete_pipeline_phase4c4 test_complete_pipeline_phase4c4.cpp)
target_link_libraries(test_complete_pipeline_phase4c4
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    pgx_logging
)
add_test(NAME CompletePipelinePhase4c4Test COMMAND test_complete_pipeline_phase4c4)

# Simple Pipeline test for debugging
add_executable(test_simple_pipeline_phase4c4 test_simple_pipeline_phase4c4.cpp)
target_link_libraries(test_simple_pipeline_phase4c4
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    pgx_logging
)
add_test(NAME SimplePipelinePhase4c4Test COMMAND test_simple_pipeline_phase4c4)

# Phase 4c-4 Basic test
add_executable(test_phase4c4_basic test_phase4c4_basic.cpp)
target_link_libraries(test_phase4c4_basic
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    pgx_logging
)
add_test(NAME Phase4c4BasicTest COMMAND test_phase4c4_basic)

# Phase 4c-4 Simplified architecture test (no segfault)
add_executable(test_relalg_to_db_phase4c4 test_relalg_to_db_phase4c4.cpp)
target_link_libraries(test_relalg_to_db_phase4c4
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    pgx_logging
)
add_test(NAME RelAlgToDBPhase4c4Test COMMAND test_relalg_to_db_phase4c4)

# Simple RelAlgToDB tests for debugging
add_executable(test_relalg_to_db_simple test_relalg_to_db_simple.cpp)
target_link_libraries(test_relalg_to_db_simple
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    pgx_logging
)
add_test(NAME RelAlgToDBSimpleTest COMMAND test_relalg_to_db_simple)

# Segfault debug test
add_executable(test_segfault_debug test_segfault_debug.cpp)
target_link_libraries(test_segfault_debug
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    pgx_logging
)
add_test(NAME SegfaultDebugTest COMMAND test_segfault_debug)

# Materialize only test
add_executable(test_materialize_only test_materialize_only.cpp)
target_link_libraries(test_materialize_only
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    pgx_logging
)
add_test(NAME MaterializeOnlyTest COMMAND test_materialize_only)

# Fixed RelAlgToDB test
add_executable(test_relalg_to_db_fixed test_relalg_to_db_fixed.cpp)
target_link_libraries(test_relalg_to_db_fixed
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRParser
    pgx_logging
)
add_test(NAME RelAlgToDBFixedTest COMMAND test_relalg_to_db_fixed)

# Minimal RelAlgToDB test (to diagnose segfault)
add_executable(test_minimal_relalg_to_db test_minimal_relalg_to_db.cpp)
target_link_libraries(test_minimal_relalg_to_db
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
    pgx_logging
)
add_test(NAME MinimalRelAlgToDBTest COMMAND test_minimal_relalg_to_db)

