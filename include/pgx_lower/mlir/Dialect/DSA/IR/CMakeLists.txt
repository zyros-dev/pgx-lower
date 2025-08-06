# DSA TableGen definitions
set(LLVM_TARGET_DEFINITIONS DSAOps.td)
mlir_tablegen(DSAOps.h.inc -gen-op-decls)
mlir_tablegen(DSAOps.cpp.inc -gen-op-defs)
mlir_tablegen(DSAOpsDialect.h.inc -gen-dialect-decls --dialect=dsa)
mlir_tablegen(DSAOpsDialect.cpp.inc -gen-dialect-defs --dialect=dsa)
mlir_tablegen(DSAOpsTypes.h.inc -gen-typedef-decls --typedefs-dialect=dsa)
mlir_tablegen(DSAOpsTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=dsa)
add_public_tablegen_target(MLIRDSAOpsIncGen)