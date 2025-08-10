# Util to LLVM conversion pass
add_mlir_dialect_library(UtilToLLVM
        LowerToLLVM.cpp
        UtilTypeConversion.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/pgx_lower/mlir/Conversion/UtilToLLVM

    DEPENDS
    MLIRUtilOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRUtilDialect
    MLIRLLVMDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLLVMCommonConversion
    MLIRTransformUtils
)