add_mlir_dialect_library(MLIRDSAToLLVMConversion
    DSAToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/pgx_lower/mlir/Conversion/DSAToLLVM

    DEPENDS
    MLIRDSAOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRDSADialect
    MLIRRelAlgDialect
    MLIRLLVMDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTransformUtils
    MLIRLLVMCommonConversion
)