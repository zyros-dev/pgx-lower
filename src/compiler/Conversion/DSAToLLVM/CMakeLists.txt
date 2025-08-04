add_mlir_conversion_library(MLIRDSAToLLVM
        LowerDSAToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDSADialect
        MLIRLLVMDialect
)

set_target_properties(MLIRDSAToLLVM PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)