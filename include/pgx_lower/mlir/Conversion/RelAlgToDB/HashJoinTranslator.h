#ifndef MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H
#include "JoinTranslator.h"
#include "Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/IRMapping.h"
#include <tuple>

namespace pgx::mlir::relalg {
class HashJoinUtils {
   public:
   static std::tuple<ColumnSet, ColumnSet, std::vector<::mlir::Type>, std::vector<ColumnSet>, std::vector<bool>> analyzeHJPred(::mlir::Block* block, ColumnSet availableLeft, ColumnSet availableRight) {
      llvm::DenseMap<mlir::Value, ColumnSet> required;
      llvm::DenseSet<::mlir::Value> pureAttribute;

      ColumnSet leftKeys, rightKeys;
      std::vector<ColumnSet> leftKeyAttributes;
      std::vector<bool> canSave;
      std::vector<::mlir::Type> types;
      block->walk([&](::mlir::Operation* op) {
         if (auto getAttr = ::mlir::dyn_cast_or_null<GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), ColumnSet::from(getAttr.attr())});
            pureAttribute.insert(getAttr.getResult());
         } else if (auto cmpOp = ::mlir::dyn_cast_or_null<CmpOpInterface>(op)) {
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
            ColumnSet attributes;
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
      if (::mlir::isa<ReturnOp>(op)) {
         return true;
      }
      if (::mlir::isa<mlir::db::AndOp>(op) || first) {
         for (auto* user : op->getUsers()) {
            if (!isAndedResult(user, false)) return false;
         }
         return true;
      } else {
         return false;
      }
   }
   static std::vector<::mlir::Value> inlineKeys(::mlir::Block* block, ColumnSet keyAttributes, ColumnSet otherAttributes, ::mlir::Block* newBlock, ::mlir::Block::iterator insertionPoint, TranslatorContext& context) {
      llvm::DenseMap<mlir::Value, ColumnSet> required;
      ::mlir::IRMapping mapping;
      std::vector<::mlir::Value> keys;
      block->walk([&](::mlir::Operation* op) {
         if (auto getAttr = ::mlir::dyn_cast_or_null<GetColumnOp>(op)) {
            required.insert({getAttr.getResult(), ColumnSet::from(getAttr.attr())});
            if (keyAttributes.intersects(ColumnSet::from(getAttr.attr()))) {
               mapping.map(getAttr.getResult(), context.getValueForAttribute(&getAttr.attr().getColumn()));
            }
         } else if (auto cmpOp = ::mlir::dyn_cast_or_null<CmpOpInterface>(op)) {
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
                     mlir::OpBuilder builder(cmpOp->getContext());
                     builder.setInsertionPoint(newBlock, insertionPoint);
                     auto helperOp = builder.create<arith::ConstantOp>(cmpOp.getLoc(), builder.getIndexAttr(0));

                     detail::inlineOpIntoBlock(keyVal.getDefiningOp(), keyVal.getDefiningOp()->getParentOp(), newBlock->getParentOp(), newBlock, mapping, helperOp);
                     helperOp->remove();
                     helperOp->destroy();
                  }
                  keys.push_back(mapping.lookupOrNull(keyVal));
               }
            }
         } else {
            ColumnSet attributes;
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

class HashJoinTranslator : public JoinTranslator {
   public:
   mlir::Location loc;
   ColumnSet leftKeys, rightKeys;
   OrderedAttributes orderedKeys;
   OrderedAttributes orderedValues;
   mlir::TupleType keyTupleType, valTupleType, entryType;
   ::mlir::Value joinHashtable;

   HashJoinTranslator(std::shared_ptr<JoinImpl> impl) : JoinTranslator(impl), loc(joinOp.getLoc()) {}

   public:
   virtual void setInfo(Translator* consumer, ColumnSet requiredAttributes) override;
   virtual void produce(TranslatorContext& context, mlir::OpBuilder& builder) override;

   void unpackValues(TranslatorContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, TranslatorContext& context, Value& marker);
   void unpackKeys(TranslatorContext::AttributeResolverScope& scope, OpBuilder& builder, Value packed, TranslatorContext& context);

   virtual void scanHT(TranslatorContext& context, mlir::OpBuilder& builder) override;
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, TranslatorContext& context) override;

   virtual ~HashJoinTranslator() {}
};
} // end namespace mlir::relalg

#endif // MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H
