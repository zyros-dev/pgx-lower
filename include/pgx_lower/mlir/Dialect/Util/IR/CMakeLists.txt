# Util TableGen definitions
set(LLVM_TARGET_DEFINITIONS UtilOps.td)
mlir_tablegen(UtilOps.h.inc -gen-op-decls)
mlir_tablegen(UtilOps.cpp.inc -gen-op-defs)
mlir_tablegen(UtilOpsDialect.h.inc -gen-dialect-decls --dialect=util)
mlir_tablegen(UtilOpsDialect.cpp.inc -gen-dialect-defs --dialect=util)
mlir_tablegen(UtilOpsTypes.h.inc -gen-typedef-decls --typedefs-dialect=util)
mlir_tablegen(UtilOpsTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=util)
add_public_tablegen_target(MLIRUtilOpsIncGen)