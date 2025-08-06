# RelAlgToDSA Conversion CMakeLists.txt
# Based on LingoDB conversion patterns

add_mlir_library(MLIRRelAlgToDSAConversion
    RelAlgToDSA.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRRelAlgDialect
    MLIRDSADialect
)