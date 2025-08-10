# RelAlg TableGen definitions
set(LLVM_TARGET_DEFINITIONS RelAlgOps.td)
mlir_tablegen(RelAlgOps.h.inc -gen-op-decls)
mlir_tablegen(RelAlgOps.cpp.inc -gen-op-defs)
mlir_tablegen(RelAlgOpsDialect.h.inc -gen-dialect-decls --dialect=relalg)
mlir_tablegen(RelAlgOpsDialect.cpp.inc -gen-dialect-defs --dialect=relalg)
mlir_tablegen(RelAlgOpsAttributes.h.inc -gen-attrdef-decls --attrdefs-dialect=relalg)
mlir_tablegen(RelAlgOpsAttributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect=relalg)
mlir_tablegen(RelAlgOpsTypes.h.inc -gen-typedef-decls --typedefs-dialect=relalg)
mlir_tablegen(RelAlgOpsTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=relalg)

# Generate interfaces from RelAlgInterfaces.td
set(LLVM_TARGET_DEFINITIONS RelAlgInterfaces.td)
mlir_tablegen(RelAlgOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(RelAlgOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRRelAlgInterfacesIncGen)

add_public_tablegen_target(MLIRRelAlgOpsIncGen)