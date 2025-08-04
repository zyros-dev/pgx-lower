add_mlir_conversion_library(MLIRUtilToLLVM
        LowerToLLVM.cpp
        UtilTypeConversion.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRUtilDialect
)
