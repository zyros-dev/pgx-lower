#ifndef MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H
#include "JoinTranslator.h"
#include "Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include <tuple>

namespace pgx::mlir::relalg {
class HashJoinUtils {
   public:
   static std::tuple<pgx::mlir::relalg::ColumnSet, pgx::mlir::relalg::ColumnSet, std::vector<::mlir::Type>, std::vector<ColumnSet>, std::vector<bool>> analyzeHJPred(::mlir::Block* block, pgx::mlir::relalg::ColumnSet availableLeft, pgx::mlir::relalg::ColumnSet availableRight) {
      llvm::DenseMap<::mlir::Value, pgx::mlir::relalg::ColumnSet> required;
      llvm::DenseSet<::mlir::Value> pureAttribute;

      pgx::mlir::relalg::ColumnSet leftKeys, rightKeys;
      std::vector<ColumnSet> leftKeyAttributes;
      std::vector<bool> canSave;
      std::vector<::mlir::Type> types;
      block->walk([&](::mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<pgx::mlir::relalg::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), pgx::mlir::relalg::ColumnSet::from(getAttr.getAttr())});
            pureAttribute.insert(getAttr.getResult());
         } else if (auto cmpOp = mlir::dyn_cast_or_null<pgx::mlir::relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred() && isAndedResult(op)) {
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  leftKeys.insert(leftAttributes);
                  rightKeys.insert(rightAttributes);
                  leftKeyAttributes.push_back(leftAttributes);
                  canSave.push_back(pureAttribute.contains(cmpOp.getLeft()));
                  types.push_back(cmpOp.getRight().getType());

               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  leftKeys.insert(rightAttributes);
                  rightKeys.insert(leftAttributes);
                  leftKeyAttributes.push_back(rightAttributes);
                  canSave.push_back(pureAttribute.contains(cmpOp.getRight()));
                  types.push_back(cmpOp.getRight().getType());
               }
            }
         } else {
            pgx::mlir::relalg::ColumnSet attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) {
               required.insert({result, attributes});
            }
         }
      });
      return {leftKeys, rightKeys, types, leftKeyAttributes, canSave};
   }

   static bool isAndedResult(::mlir::Operation* op, bool first = true) {
      if (mlir::isa<pgx::mlir::relalg::ReturnOp>(op)) {
         return true;
      }
      if (mlir::isa<pgx::mlir::db::AndOp>(op) || first) {
         for (auto* user : op->getUsers()) {
            if (!isAndedResult(user, false)) return false;
         }
         return true;
      } else {
         return false;
      }
   }
   static std::vector<::mlir::Value> inlineKeys(::mlir::Block* block, pgx::mlir::relalg::ColumnSet keyAttributes,pgx::mlir::relalg::ColumnSet otherAttributes, ::mlir::Block* newBlock, ::mlir::Block::iterator insertionPoint, pgx::mlir::relalg::TranslatorContext& context) {
      llvm::DenseMap<::mlir::Value, pgx::mlir::relalg::ColumnSet> required;
      ::mlir::BlockAndValueMapping mapping;
      std::vector<::mlir::Value> keys;
      block->walk([&](::mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<pgx::mlir::relalg::GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), pgx::mlir::relalg::ColumnSet::from(getAttr.getAttr())});
            if (keyAttributes.intersects(pgx::mlir::relalg::ColumnSet::from(getAttr.getAttr()))) {
               mapping.map(getAttr.getResult(), context.getValueForAttribute(&getAttr.getAttr().getColumn()));
            }
         } else if (auto cmpOp = mlir::dyn_cast_or_null<pgx::mlir::relalg::CmpOpInterface>(op)) {
            if (cmpOp.isEqualityPred() && isAndedResult(op)) {
               auto leftAttributes = required[cmpOp.getLeft()];
               auto rightAttributes = required[cmpOp.getRight()];
               ::mlir::Value keyVal;
               if (leftAttributes.isSubsetOf(keyAttributes)&&rightAttributes.isSubsetOf(otherAttributes)) {
                  keyVal = cmpOp.getLeft();
               } else if (rightAttributes.isSubsetOf(keyAttributes)&&leftAttributes.isSubsetOf(otherAttributes)) {
                  keyVal = cmpOp.getRight();
               }
               if (keyVal) {
                  if (!mapping.contains(keyVal)) {
                     //todo: remove nasty hack:
                     ::mlir::OpBuilder builder(cmpOp->getContext());
                     builder.setInsertionPoint(newBlock, insertionPoint);
                     auto helperOp = builder.create<arith::ConstantOp>(cmpOp.getLoc(), builder.getIndexAttr(0));

                     pgx::mlir::relalg::detail::inlineOpIntoBlock(keyVal.getDefiningOp(), keyVal.getDefiningOp()->getParentOp(), newBlock->getParentOp(), newBlock, mapping, helperOp);
                     helperOp->remove();
                     helperOp->destroy();
                  }
                  keys.push_back(mapping.lookupOrNull(keyVal));
               }
            }
         } else {
            pgx::mlir::relalg::ColumnSet attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) {
               required.insert({result, attributes});
            }
         }
      });
      return keys;
   }
};

class HashJoinTranslator : public pgx::mlir::relalg::JoinTranslator {
   public:
   ::mlir::Location loc;
   pgx::mlir::relalg::ColumnSet leftKeys, rightKeys;
   pgx::mlir::relalg::OrderedAttributes orderedKeys;
   pgx::mlir::relalg::OrderedAttributes orderedValues;
   mlir::TupleType keyTupleType, valTupleType, entryType;
   ::mlir::Value joinHashtable;

   HashJoinTranslator(std::shared_ptr<JoinImpl> impl) : JoinTranslator(impl), loc(joinOp.getLoc()) {}

   public:
   virtual void setInfo(pgx::mlir::relalg::Translator* consumer, pgx::mlir::relalg::ColumnSet requiredAttributes) override;
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override;

   void unpackValues(TranslatorContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, TranslatorContext& context, Value& marker);
   void unpackKeys(TranslatorContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, TranslatorContext& context);

   virtual void scanHT(TranslatorContext& context, ::mlir::OpBuilder& builder) override;
   virtual void consume(pgx::mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override;

   virtual ~HashJoinTranslator() {}
};
} // end namespace pgx::mlir::relalg

#endif // MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H
