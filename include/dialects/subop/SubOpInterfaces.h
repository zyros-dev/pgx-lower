#ifndef SUBOP_INTERFACES_H
#define SUBOP_INTERFACES_H
// #include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h" // TODO: Port
// #include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h" // TODO: Port
// #include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h" // TODO: Port
#include "dialects/tuplestream/Column.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <string>
#include <vector>

namespace mlir::subop {
// TODO: Adapt for PostgreSQL tuple-oriented approach (vs LingoDB column-oriented)
class ColumnMapping {
   std::unordered_map<mlir::tuples::Column*, mlir::tuples::Column*> mapping;

   public:
   // TODO: Implement tuple-oriented remapping for PostgreSQL
   // Stub implementations for now
   mlir::Attribute remap(mlir::Attribute attr) {
      // TODO: Implement proper remapping
      return attr;
   }
   // TODO: Implement array/dictionary remapping and cloning for PostgreSQL
   void mapRaw(mlir::tuples::Column* from, mlir::tuples::Column* to) {
      mapping[from] = to;
   }
};
// Minimal stub - TODO: Implement for PostgreSQL state management  
class SubOpStateUsageTransformer {
public:
   // Stub implementation
};

} // end namespace mlir::subop
// #define GET_OP_CLASSES
// #include "SubOpOpsInterfaces.h.inc" // TODO: Generate this file if needed

#endif //SUBOP_INTERFACES_H
