#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H

#include "mlir/Conversion/RelAlgToDB/JoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
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
   static std::tuple<ColumnSet, ColumnSet, std::vector<mlir::Type>, std::vector<ColumnSet>, std::vector<bool>> 
   analyzeHJPred(mlir::Block* block, ColumnSet availableLeft, ColumnSet availableRight);
   
   static bool isAndedResult(mlir::Operation* op, bool first = true);
   
   static std::vector<mlir::Value> inlineKeys(mlir::Block* block, ColumnSet keyAttributes,
                                               ColumnSet otherAttributes, mlir::Block* newBlock,
                                               mlir::Block::iterator insertionPoint, 
                                               TranslatorContext& context);
};

class HashJoinTranslator : public JoinTranslator {
   public:
   mlir::Location loc;
   ColumnSet leftKeys, rightKeys;
   OrderedAttributes orderedKeys;
   OrderedAttributes orderedValues;
   mlir::TupleType keyTupleType, valTupleType, entryType;
   mlir::Value joinHashtable;
   
   HashJoinTranslator(std::shared_ptr<JoinImpl> impl);
   
   virtual void setInfo(Translator* consumer, ColumnSet requiredAttributes) override;
   virtual void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override;
   
   void unpackValues(TranslatorContext::AttributeResolverScope& scope, ::mlir::OpBuilder& builder, 
                     ::mlir::Value packed, TranslatorContext& context, ::mlir::Value& marker);
   void unpackKeys(TranslatorContext::AttributeResolverScope& scope, ::mlir::OpBuilder& builder, 
                   ::mlir::Value packed, TranslatorContext& context);
   
   virtual void scanHT(TranslatorContext& context, ::mlir::OpBuilder& builder) override;
   virtual void consume(Translator* child, ::mlir::OpBuilder& builder, TranslatorContext& context) override;
   
   virtual ~HashJoinTranslator() = default;
};

} // namespace pgx::mlir::relalg

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_HASHJOINTRANSLATOR_H