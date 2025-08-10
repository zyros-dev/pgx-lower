#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H

#include "mlir/Conversion/RelAlgToDB/JoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

namespace pgx::mlir::relalg {

class NLJoinTranslator : public JoinTranslator {
   OrderedAttributes orderedAttributesLeft;
   mlir::TupleType tupleTypeLeft;
   mlir::Value vector;
   mlir::Location loc;
   
   public:
   NLJoinTranslator(std::shared_ptr<JoinImpl> impl);
   
   virtual void setInfo(Translator* consumer, ColumnSet requiredAttributes) override;
   virtual void build(mlir::OpBuilder& builder, TranslatorContext& context);
   virtual void scanHT(TranslatorContext& context, mlir::OpBuilder& builder) override;
   virtual void produce(TranslatorContext& context, mlir::OpBuilder& builder) override;
   virtual void consume(Translator* child, mlir::OpBuilder& builder, TranslatorContext& context) override;
   
   virtual ~NLJoinTranslator() = default;
};

} // namespace pgx::mlir::relalg

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H