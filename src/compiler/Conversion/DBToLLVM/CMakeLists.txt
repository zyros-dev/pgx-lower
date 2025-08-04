add_mlir_conversion_library(MLIRDBToLLVM
        LowerDBToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDBDialect
        MLIRLLVMDialect
)

set_target_properties(MLIRDBToLLVM PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)