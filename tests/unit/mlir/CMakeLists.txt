# Unit tests for MLIR pipeline integration

add_executable(test_passes_integration test_passes_integration.cpp)

target_link_libraries(test_passes_integration
    gtest gtest_main
    MLIRContext
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRMemRefDialect
    RelAlgDialect
    DBDialect
    DSADialect
    UtilDialect
    MLIRPasses
    pgx_lower_static
)

add_test(NAME PassesIntegrationTest COMMAND test_passes_integration)

# Unit tests for sequential pipeline approach
add_executable(test_sequential_pipelines test_sequential_pipelines.cpp)

target_link_libraries(test_sequential_pipelines
    gtest gtest_main
    MLIRContext
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRMemRefDialect
    RelAlgDialect
    DBDialect
    DSADialect
    UtilDialect
    MLIRPasses
    StandardToLLVM
    MLIRDBToArrowStd
    MLIRDSAToStd
    pgx_lower_static
)

add_test(NAME SequentialPipelinesTest COMMAND test_sequential_pipelines)