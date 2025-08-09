# MLIR Pass Pipeline Infrastructure CMakeLists.txt
# Centralized pass registration and pipeline configuration for pgx-lower
# Architecture: PostgreSQL AST → RelAlg → (DB + DSA + Util) → Standard MLIR → LLVM IR → JIT

add_mlir_library(MLIRPgxLowerPasses
    Passes.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRRelAlgDialect
    MLIRDBDialect
    MLIRDSADialect
    MLIRRelAlgToDBConversion
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLLVMDialect
)