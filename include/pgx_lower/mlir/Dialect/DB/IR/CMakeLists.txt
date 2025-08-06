# DB TableGen definitions
set(LLVM_TARGET_DEFINITIONS DBOps.td)
mlir_tablegen(DBOps.h.inc -gen-op-decls)
mlir_tablegen(DBOps.cpp.inc -gen-op-defs)
mlir_tablegen(DBOpsDialect.h.inc -gen-dialect-decls --dialect=db)
mlir_tablegen(DBOpsDialect.cpp.inc -gen-dialect-defs --dialect=db)
mlir_tablegen(DBOpsTypes.h.inc -gen-typedef-decls --typedefs-dialect=db)
mlir_tablegen(DBOpsTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=db)
add_public_tablegen_target(MLIRDBOpsIncGen)