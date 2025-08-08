# RelAlgToDB Conversion CMakeLists.txt
# Implements first stage of LingoDB pipeline: RelAlg -> DB using Translator pattern

add_mlir_library(MLIRRelAlgToDBConversion
    RelAlgToDB.cpp
    Translator.cpp
    TranslatorContext.cpp
    BaseTableTranslator.cpp
    MaterializeTranslator.cpp

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