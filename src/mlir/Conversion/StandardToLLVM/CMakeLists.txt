add_library(StandardToLLVM
    StandardToLLVM.cpp
)

target_link_libraries(StandardToLLVM PUBLIC
    MLIRIR
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRTransforms
    MLIRPass
    MLIRSupport
    MLIRSCFToControlFlow
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRMemRefToLLVM
    MLIRUtilToLLVM
    MLIRAffineToStandard
    mlir-support
)