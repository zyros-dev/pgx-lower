#ifndef SUBOP_INTERFACES_H
#define SUBOP_INTERFACES_H

// IMPORTANT: This file contains STUB IMPLEMENTATIONS to get pgx-lower compiling.
// These are minimal placeholders for LingoDB's column-oriented data structures.
// 
// PostgreSQL is tuple-oriented, not column-oriented like LingoDB/PyArrow.
// These stubs need to be properly implemented to handle PostgreSQL's row-based
// data model when we actually implement the lowering passes.
//
// The cloneSubOp() methods referenced in SubOpOps.td are expected to exist
// but are not implemented yet - they're just declarations in the generated code.
// #include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h" // TODO Phase 5: Port
// #include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h" // TODO Phase 5: Port
// #include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h" // TODO Phase 5: Port
#include "compiler/Dialect/TupleStream/Column.h"
#include "compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <string>
#include <vector>

// Our implementations in a clear namespace
namespace pgx_lower::compiler::dialect::subop {

// Forward declaration of attribute types
class StateMembersAttr;
// PGX_LOWER STUB IMPLEMENTATION - TODO Phase 5: Adapt for PostgreSQL tuple-oriented approach
// This is a minimal stub to get compilation working. LingoDB uses this for column-oriented
// data remapping, but PostgreSQL is tuple-oriented.
class ColumnMapping {
   std::unordered_map<pgx_lower::compiler::dialect::tuples::Column*, 
                      pgx_lower::compiler::dialect::tuples::Column*> mapping;

   public:
   // TODO Phase 5: Implement tuple-oriented remapping for PostgreSQL
   // Stub implementations for now
   mlir::Attribute remap(mlir::Attribute attr) {
      // TODO Phase 5: Implement proper remapping
      return attr;
   }
   // TODO Phase 5: Implement array/dictionary remapping and cloning for PostgreSQL
   void mapRaw(pgx_lower::compiler::dialect::tuples::Column* from, 
               pgx_lower::compiler::dialect::tuples::Column* to) {
      mapping[from] = to;
   }
};
// PGX_LOWER - SubOpStateUsageTransformer is defined in StateUsageTransformer.h
// But generated code expects it here, so we forward declare
class ColumnUsageAnalysis;
class SubOpStateUsageTransformer;

} // end namespace pgx_lower::compiler::dialect::subop

// Include generated interface definitions
#include "SubOpOpsInterfaces.h.inc"

#endif //SUBOP_INTERFACES_H
