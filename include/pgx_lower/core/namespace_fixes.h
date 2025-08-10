#ifndef PGX_CORE_NAMESPACE_FIXES_H
#define PGX_CORE_NAMESPACE_FIXES_H

// This header provides namespace fixes for LLVM 20 compatibility
// When TableGen generates code in custom namespaces, it doesn't properly
// qualify MLIR types, leading to compilation errors.

// Import the core MLIR types into the pgx::mlir namespace
namespace pgx::mlir {
  using ::mlir::Type;
  using ::mlir::Attribute;
  using ::mlir::Value;
  using ::mlir::MLIRContext;
  using ::mlir::Operation;
  using ::mlir::OpBuilder;
  using ::mlir::Location;
  using ::mlir::ValueRange;
  using ::mlir::ResultRange;
  using ::mlir::ModuleOp;
  using ::mlir::AsmParser;
  using ::mlir::AsmPrinter;
  using ::mlir::TupleType;
} // namespace pgx::mlir

#endif // PGX_CORE_NAMESPACE_FIXES_H