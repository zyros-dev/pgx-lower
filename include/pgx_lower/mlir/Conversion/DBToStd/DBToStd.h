//===- DBToStd.h - DB to Standard dialects conversion pass -------*- C++ -*-===//

#ifndef MLIR_CONVERSION_DBTOSTD_DBTOSTD_H
#define MLIR_CONVERSION_DBTOSTD_DBTOSTD_H

#include <memory>

namespace mlir {
class Pass;

/// Create a pass to convert DB dialect operations to Standard MLIR dialects
/// with PostgreSQL SPI function calls (Phase 4d).
/// 
/// Converts DB operations from PostgreSQL table access to actual SPI calls:
/// - db.get_external → func.call @pg_table_open
/// - db.iterate_external → func.call @pg_get_next_tuple  
/// - db.get_field → func.call @pg_extract_field
/// - db.nullable_get_val → llvm.extractvalue
/// - Pure arithmetic → Standard MLIR (arith.addi, arith.cmpi)
std::unique_ptr<Pass> createDBToStdPass();

} // namespace mlir

#endif // MLIR_CONVERSION_DBTOSTD_DBTOSTD_H