add_mlir_dialect_library(MLIRUtilDialect
    UtilDialect.cpp
    UtilOps.cpp
    UtilTypes.cpp
    FunctionHelper.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/pgx_lower/mlir/Dialect/Util

    DEPENDS
    MLIRUtilOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)