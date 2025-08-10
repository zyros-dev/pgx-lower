# DB Dialect IR CMakeLists.txt
# Based on LingoDB patterns

add_mlir_dialect_library(MLIRDBDialect
    DBDialect.cpp
    DBOps.cpp
    DBTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/include/pgx_lower/mlir/Dialect/DB

    DEPENDS
    MLIRDBOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
)