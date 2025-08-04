#ifndef PGX_LOWER_COMPILER_DIALECT_SUBOPERATOR_UTILS_H
#define PGX_LOWER_COMPILER_DIALECT_SUBOPERATOR_UTILS_H
#include "compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <unordered_map>
namespace pgx_lower::compiler::dialect::subop {
class MapCreationHelper {
   std::unordered_map<pgx_lower::compiler::dialect::tuples::Column*, size_t> columnToIndex;
   std::vector<mlir::Attribute> colRefs;
   mlir::Block* mapBlock;
   mlir::MLIRContext* context;

   public:
   MapCreationHelper(mlir::MLIRContext* context) : context(context) {
      mapBlock = new mlir::Block;
   }
   mlir::Value access(pgx_lower::compiler::dialect::tuples::ColumnRefAttr columnRefAttr, mlir::Location loc) {
      // TODO Phase 5: Fix when proper Column management is implemented
      auto& keyColumn = columnRefAttr.getColumn();
      if (columnToIndex.contains(&keyColumn)) {
         return mapBlock->getArgument(columnToIndex[&keyColumn]);
      } else {
         auto arg = mapBlock->addArgument(keyColumn.type, loc);
         columnToIndex[&keyColumn] = columnToIndex.size();
         colRefs.push_back(columnRefAttr);
         return arg;
      }
   }
   mlir::ArrayAttr getColRefs() {
      return mlir::ArrayAttr::get(context, colRefs);
   }
   template <class B, class F>
   void buildBlock(B& builder, const F& f) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(mapBlock);
      f(builder);
   }
   mlir::Block* getMapBlock() {
      return mapBlock;
   }
};
} //end namespace pgx_lower::compiler::dialect::subop

#endif //PGX_LOWER_COMPILER_DIALECT_SUBOPERATOR_UTILS_H
