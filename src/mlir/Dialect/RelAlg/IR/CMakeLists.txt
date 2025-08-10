# RelAlg Dialect IR CMakeLists.txt
# Based on LingoDB patterns

add_mlir_dialect_library(MLIRRelAlgDialect
    RelAlgDialect.cpp
    RelAlgOps.cpp
    RelAlgTypes.cpp
    ColumnManager.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/mlir/Dialect/RelAlg

    DEPENDS
    MLIRRelAlgOpsIncGen
    MLIRRelAlgInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
)