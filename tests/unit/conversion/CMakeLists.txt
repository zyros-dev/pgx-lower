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

# RelAlg → DB conversion integration tests
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

# Register RelAlg → DB integration tests with CTest
add_test(NAME Phase3AIntegrationTest COMMAND test_phase3a_integration)

# NOTE: test_functional_conversions.cpp deleted - obsolete conversion tests

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

# Phase 4e: DSA → LLVM conversion tests (data structure algorithms in parallel lowering)
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

# NOTE: test_relalg_to_db.cpp deleted - obsolete tests expecting old DB operations

# NOTE: test_materialize_phase4c2.cpp deleted - obsolete tests from legacy Phase 4c-2

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

# BaseTable streaming tests (PostgreSQL table access with constant memory)
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

# Streaming memory validation tests (constant memory usage verification)
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

# Complete Pipeline Integration tests (RelAlg → DB → DSA streaming)
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

# RelAlgToDB fixes test
add_executable(test_relalg_to_db_fixes test_relalg_to_db_fixes.cpp)
target_link_libraries(test_relalg_to_db_fixes
    gtest
    gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRPass
    MLIRIR
    MLIRSupport
    PostgresQueryLogger
)
add_test(NAME RelAlgToDBFixesTest COMMAND test_relalg_to_db_fixes)

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

# NOTE: test_phase4c4_basic.cpp removed - obsolete test with old BaseTableOp syntax

# Complete pipeline simplified test (memory safety validation)
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

# NOTE: test_minimal_relalg_to_db.cpp removed - obsolete test with old BaseTableOp syntax

# BaseTable DSA streaming pattern test
add_executable(test_base_table_dsa relalg_to_db/test_base_table_dsa.cpp)
target_link_libraries(test_base_table_dsa
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME BaseTableDSATest COMMAND test_base_table_dsa)

# Phase 4d: DB → Standard MLIR conversion tests (PostgreSQL SPI integration)
add_executable(test_db_to_std_basic db_to_std/test_db_to_std_basic.cpp)
target_link_libraries(test_db_to_std_basic
    gtest gtest_main
    MLIRDBDialect
    MLIRDSADialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRDBToStdConversion
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME DBToStdBasicTest COMMAND test_db_to_std_basic)

# Phase 4d: PostgreSQL SPI calls test - Validate SPI function call generation
add_executable(test_postgresql_spi_calls db_to_std/test_postgresql_spi_calls.cpp)
target_link_libraries(test_postgresql_spi_calls
    gtest gtest_main
    MLIRDBDialect
    MLIRDSADialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRDBToStdConversion
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME PostgreSQLSPICallsTest COMMAND test_postgresql_spi_calls)

# DBToStd edge case tests - Validate robustness and error handling
add_executable(test_dbtostd_edge_cases db_to_std/test_dbtostd_edge_cases.cpp)
target_link_libraries(test_dbtostd_edge_cases
    gtest gtest_main
    MLIRDBDialect
    MLIRDSADialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRDBToStdConversion
    MLIRDialectConversion
    MLIRTransforms
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME DBToStdEdgeCasesTest COMMAND test_dbtostd_edge_cases)

# Hybrid RelAlgToDB test - Verify DSA+PostgreSQL SPI architecture
add_executable(test_hybrid_relalgto_db test_hybrid_relalgto_db.cpp)
target_link_libraries(test_hybrid_relalgto_db
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME HybridRelAlgToDBTest COMMAND test_hybrid_relalgto_db)

# Phase 4d-6: Complete Pipeline Integration tests (RelAlg → DB+DSA → Standard MLIR → LLVM)
add_executable(test_pipeline_integration test_pipeline_integration.cpp)
target_link_libraries(test_pipeline_integration
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    # MLIRDBToStdConversion # Not implemented yet
    MLIRDSAToLLVMConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME PipelineIntegrationTest COMMAND test_pipeline_integration)

# Phase 4e-1: Column Resolution Architecture Fix tests
add_executable(test_column_resolution_fix test_column_resolution_fix.cpp)
target_link_libraries(test_column_resolution_fix
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME ColumnResolutionFixTest COMMAND test_column_resolution_fix)

# LingoDB Column Architecture tests - Verify column identity sharing
add_executable(test_lingodb_column_architecture test_lingodb_column_architecture.cpp)
target_link_libraries(test_lingodb_column_architecture
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME LingoDBColumnArchitectureTest COMMAND test_lingodb_column_architecture)

# Consumer Chain Fix tests - Verify MaterializeTranslator::consume() execution
add_executable(test_consumer_chain_fixed test_consumer_chain_fixed.cpp)
target_link_libraries(test_consumer_chain_fixed
    gtest gtest_main
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRPass
    MLIRIR
    MLIRSupport
    pgx_logging
)
add_test(NAME ConsumerChainFixedTest COMMAND test_consumer_chain_fixed)

