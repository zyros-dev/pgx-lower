# DBToDSA Conversion CMakeLists.txt  
# Implements second stage of LingoDB pipeline: DB -> DSA

add_mlir_library(MLIRDBToDSAConversion
    DBToDSA.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRDBDialect
    MLIRDSADialect
)