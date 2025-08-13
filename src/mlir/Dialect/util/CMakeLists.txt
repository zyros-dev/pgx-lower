add_mlir_dialect_library(MLIRUtilDialect
        UtilDialect.cpp
        UtilOps.cpp
        UtilTypes.cpp
        FunctionHelper.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/util

        DEPENDS
        MLIRUtilOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
