add_library(MLIRPasses
    Passes.cpp
)

target_link_libraries(MLIRPasses
    PUBLIC
    MLIRIR
    MLIRPass
    MLIRRelAlgToDB
    MLIRDBToArrowStd
    MLIRDSAToStd
    MLIRTransforms
    MLIRSCFToControlFlow
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts
)