#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include <lingodb/mlir/Conversion/RelAlgToDB/HashJoinTranslator.h>
#include <lingodb/mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include "pgx-lower/utility/logging.h"

using namespace ::mlir::relalg;
std::vector<::mlir::Value> mlir::relalg::Translator::mergeRelationalBlock(::mlir::Block* dest, ::mlir::Operation* op, mlir::function_ref<::mlir::Block*(::mlir::Operation*)> getBlockFn, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope) {
   // Splice the operations of the 'source' block into the 'dest' block and erase it.
   llvm::iplist<::mlir::Operation> translated;
   std::vector<::mlir::Operation*> toErase;
   auto* cloned = op->clone();
   ::mlir::Block* source = getBlockFn(cloned);
   auto* terminator = source->getTerminator();

   source->walk([&](mlir::relalg::GetColumnOp getColumnOp) {
      getColumnOp.replaceAllUsesWith(context.getValueForAttribute(&getColumnOp.getAttr().getColumn()));
      toErase.push_back(getColumnOp.getOperation());
   });
   /*for (auto addColumnOp : source->getOps<mlir::relalg::AddColumnOp>()) {
      context.setValueForAttribute(scope, &addColumnOp.getAttr().getColumn(), addColumnOp.getVal());
      toErase.push_back(addColumnOp.getOperation());
   }*/

   dest->getOperations().splice(dest->end(), source->getOperations());
   for (auto* op : toErase) {
      op->dropAllUses();
      op->erase();
   }

   // PGX-LOWER: Enforce nullability in joins earlier, which causes this operaiton
   // to require adjusting the logical joins to adjust for nullability.
   std::vector<std::pair<::mlir::Operation*, ::mlir::Operation*>> toReplace;
   for (auto& operation : dest->getOperations()) {
      if (auto inferOp = mlir::dyn_cast<mlir::InferTypeOpInterface>(&operation)) {
         PGX_LOG(RELALG_LOWER, DEBUG, "Checking InferTypeOpInterface operation: %s with %d operands",
            operation.getName().getStringRef().str().c_str(),
            (int)operation.getNumOperands());

         // Log operand types
         for (unsigned i = 0; i < operation.getNumOperands(); ++i) {
            bool isNullable = operation.getOperand(i).getType().isa<mlir::db::NullableType>();
            PGX_LOG(RELALG_LOWER, DEBUG, "  Operand %d: %s", i, isNullable ? "nullable" : "non-nullable");
         }

         llvm::SmallVector<mlir::Type, 4> inferredTypes;
         if (mlir::succeeded(inferOp.inferReturnTypes(
               operation.getContext(), operation.getLoc(), operation.getOperands(),
               operation.getAttrDictionary(), operation.getPropertiesStorage(),
               operation.getRegions(), inferredTypes))) {

            bool needsRecreate = false;
            auto results = operation.getResults();
            if (inferredTypes.size() == results.size()) {
               for (size_t i = 0; i < inferredTypes.size(); ++i) {
                  if (inferredTypes[i] != results[i].getType()) {
                     PGX_LOG(RELALG_LOWER, DEBUG, "  Result %zu: type mismatch detected", i);
                     needsRecreate = true;
                     break;
                  }
               }
            }

            if (!needsRecreate) {
               PGX_LOG(RELALG_LOWER, DEBUG, "  Types match, no recreation needed");
            }

            if (needsRecreate) {
               PGX_LOG(RELALG_LOWER, DEBUG, "needsRecreate for %s - type mismatch",
                  operation.getName().getStringRef().str().c_str());

               mlir::OpBuilder builder(dest, mlir::Block::iterator(&operation));

               // Use TypeSwitch to handle operations with InferTypeOpInterface using typed builders
               mlir::Operation* newOp = llvm::TypeSwitch<mlir::Operation*, mlir::Operation*>(&operation)
                  .Case<mlir::db::AndOp>([&](auto op) {
                     PGX_LOG(RELALG_LOWER, DEBUG, "Using typed builder for db.and");
                     return builder.create<mlir::db::AndOp>(op.getLoc(), op.getVals()).getOperation();
                  })
                  .Case<mlir::db::OrOp>([&](auto op) {
                     PGX_LOG(RELALG_LOWER, DEBUG, "Using typed builder for db.or");
                     return builder.create<mlir::db::OrOp>(op.getLoc(), op.getVals()).getOperation();
                  })
                  .Case<mlir::db::CmpOp>([&](auto op) {
                     PGX_LOG(RELALG_LOWER, DEBUG, "Using typed builder for db.compare");
                     return builder.create<mlir::db::CmpOp>(
                        op.getLoc(), op.getPredicateAttr(), op.getLeft(), op.getRight()).getOperation();
                  })
                  .Case<mlir::db::BetweenOp>([&](auto op) {
                     PGX_LOG(RELALG_LOWER, DEBUG, "Using typed builder for db.between");
                     return builder.create<mlir::db::BetweenOp>(
                        op.getLoc(), op.getVal(), op.getLower(), op.getUpper(),
                        op.getLowerInclusiveAttr(), op.getUpperInclusiveAttr()).getOperation();
                  })
                  .Case<mlir::db::OneOfOp>([&](auto op) {
                     PGX_LOG(RELALG_LOWER, DEBUG, "Using typed builder for db.oneof");
                     return builder.create<mlir::db::OneOfOp>(
                        op.getLoc(), op.getVal(), op.getVals()).getOperation();
                  })
                  .Default([&](mlir::Operation* op) {
                     PGX_LOG(RELALG_LOWER, DEBUG, "Using OperationState for %s",
                        op->getName().getStringRef().str().c_str());
                     // Fallback: Use OperationState for operations without InferTypeOpInterface
                     mlir::OperationState state(op->getLoc(), op->getName());
                     state.addOperands(op->getOperands());
                     state.addTypes(inferredTypes);
                     // Convert DictionaryAttr to ArrayRef<NamedAttribute>
                     llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
                     for (auto attr : op->getAttrDictionary())
                        attrs.push_back(attr);
                     state.addAttributes(attrs);
                     for (auto& region : op->getRegions()) {
                        auto* newRegion = state.addRegion();
                        newRegion->takeBody(region);
                     }
                     return builder.create(state);
                  });

               PGX_LOG(RELALG_LOWER, DEBUG, "Created new operation: %s",
                  newOp->getName().getStringRef().str().c_str());

               toReplace.push_back({&operation, newOp});
            }
         }
      }
   }

   for (auto [oldOp, newOp] : toReplace) {
      oldOp->replaceAllUsesWith(newOp);
      oldOp->erase();
   }

   auto returnOp = mlir::cast<mlir::relalg::ReturnOp>(terminator);
   std::vector<Value> res(returnOp.getResults().begin(), returnOp.getResults().end());
   terminator->erase();
   return res;
}

void Translator::propagateInfo() {
   for (auto& c : children) {
      auto available = c->getAvailableColumns();
      mlir::relalg::ColumnSet toPropagate = requiredAttributes.intersect(available);
      c->setInfo(this, toPropagate);
   }
}
mlir::relalg::Translator::Translator(Operator op) : op(op) {
   PGX_LOG(RELALG_LOWER, DEBUG, "Translator: Base constructor called with Operator");
   
   if (!op) {
      PGX_LOG(RELALG_LOWER, DEBUG, "Translator: Operator is null, skipping children initialization");
      return;
   }
   
   PGX_LOG(RELALG_LOWER, DEBUG, "Translator: Getting children from operator");
   for (auto child : op.getChildren()) {
      PGX_LOG(RELALG_LOWER, DEBUG, "Translator: Creating translator for child operation");
      children.push_back(mlir::relalg::Translator::createTranslator(child.getOperation()));
   }
   
   PGX_LOG(RELALG_LOWER, DEBUG, "Translator: Base constructor completed, %zu children", children.size());
}

mlir::relalg::Translator::Translator(::mlir::ValueRange potentialChildren) : op() {
   for (auto child : potentialChildren) {
      if (child.getType().isa<mlir::relalg::TupleStreamType>()) {
         children.push_back(mlir::relalg::Translator::createTranslator(child.getDefiningOp()));
      }
   }
}

void Translator::setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) {
   this->consumer = consumer;
   this->requiredAttributes = requiredAttributes;
   if (op) {
      this->requiredAttributes.insert(op.getUsedColumns());
      propagateInfo();
   }
}
mlir::relalg::ColumnSet Translator::getAvailableColumns() {
   return op.getAvailableColumns();
};

std::unique_ptr<mlir::relalg::Translator> Translator::createTranslator(::mlir::Operation* operation) {
   PGX_LOG(RELALG_LOWER, DEBUG, "Translator::createTranslator called for operation: %s", operation->getName().getStringRef().str().c_str());
   
   return ::llvm::TypeSwitch<::mlir::Operation*, std::unique_ptr<mlir::relalg::Translator>>(operation)
      .Case<BaseTableOp>([&](auto x) { 
         PGX_LOG(RELALG_LOWER, DEBUG, "Translator::createTranslator matched BaseTableOp");
         return createBaseTableTranslator(x); 
      })
      .Case<ConstRelationOp>([&](auto x) { 
         PGX_LOG(RELALG_LOWER, DEBUG, "Translator::createTranslator matched ConstRelationOp");
         return createConstRelTranslator(x); 
      })
      .Case<MaterializeOp>([&](auto x) { 
         PGX_LOG(RELALG_LOWER, DEBUG, "Translator::createTranslator matched MaterializeOp");
         return createMaterializeTranslator(x); 
      })
      .Case<SelectionOp>([&](auto x) { return createSelectionTranslator(x); })
      .Case<MapOp>([&](auto x) { return createMapTranslator(x); })
      .Case<CrossProductOp, InnerJoinOp, SemiJoinOp, AntiSemiJoinOp, OuterJoinOp, SingleJoinOp, MarkJoinOp, CollectionJoinOp>([&](::mlir::Operation* op) { return createJoinTranslator(op); })
      .Case<SortOp>([&](auto x) { return createSortTranslator(x); })
      .Case<AggregationOp>([&](auto x) { return createAggregationTranslator(x); })
      .Case<RenamingOp>([&](auto x) { return createRenamingTranslator(x); })
      .Case<ProjectionOp>([&](auto x) { return createProjectionTranslator(x); })
      .Case<LimitOp>([&](auto x) { return createLimitTranslator(x); })
      .Case<TmpOp>([&](auto x) { return createTmpTranslator(x); })
      .Case<UnionOp, IntersectOp, ExceptOp>([&](auto x) { return createSetOpTranslator(x); })
      .Default([](auto x) { 
         PGX_ERROR("Translator::createTranslator unsupported operation: %s", x->getName().getStringRef().str().c_str());
         x->emitError("Unsupported operation in RelAlg to DB lowering: ") << x->getName();
         return std::unique_ptr<Translator>(); 
      });
}
