# MLIR Pass Pipeline Infrastructure CMakeLists.txt
# Provides centralized pass registration and pipeline configuration

add_mlir_library(MLIRPgxLowerPasses
    Passes.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRDBToDSAConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
)