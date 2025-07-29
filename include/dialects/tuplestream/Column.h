#ifndef LINGODB_COMPILER_DIALECT_TUPLESTREAM_COLUMN_H
#define LINGODB_COMPILER_DIALECT_TUPLESTREAM_COLUMN_H
#include <mlir/IR/Types.h>
namespace pgx_lower::compiler::dialect::tuples {
struct Column {
   mlir::Type type;
   size_t oid = 0;  // PostgreSQL Object ID
   bool directMapping = false;
   size_t mappingPos = 0;
};
} // namespace pgx_lower::compiler::dialect::tuples

#endif //LINGODB_COMPILER_DIALECT_TUPLESTREAM_COLUMN_H
