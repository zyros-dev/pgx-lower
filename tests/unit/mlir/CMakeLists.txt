# Unit tests for MLIR pass pipeline infrastructure

add_executable(test_pass_pipeline test_pass_pipeline.cpp)
target_link_libraries(test_pass_pipeline
    gtest gtest_main
    MLIRPgxLowerPasses
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRDBToDSAConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRPass
    MLIRIR
)
add_test(NAME PassPipelineTest COMMAND test_pass_pipeline)