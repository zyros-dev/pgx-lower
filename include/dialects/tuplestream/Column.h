#ifndef TUPLESTREAM_COLUMN_H
#define TUPLESTREAM_COLUMN_H
#include <mlir/IR/Types.h>
namespace mlir::tuples {
// TODO: This is adapted from LingoDB's column-oriented design
// Need to rethink for PostgreSQL's tuple-oriented approach
struct Column {
   mlir::Type type;
   // TODO: Add PostgreSQL-specific tuple field metadata
};
} // namespace mlir::tuples

#endif //TUPLESTREAM_COLUMN_H