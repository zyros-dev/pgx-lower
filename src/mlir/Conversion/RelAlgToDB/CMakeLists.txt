# RelAlgToDB Conversion CMakeLists.txt
# Implements first stage of LingoDB pipeline: RelAlg -> DB

add_mlir_library(MLIRRelAlgToDBConversion
    RelAlgToDB.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
)