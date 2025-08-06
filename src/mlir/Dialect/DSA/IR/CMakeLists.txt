# DSA Dialect IR CMakeLists.txt
# Based on LingoDB patterns

add_mlir_dialect_library(MLIRDSADialect
    DSADialect.cpp
    DSAOps.cpp
    DSATypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/mlir/Dialect/DSA

    DEPENDS
    MLIRDSAOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
)