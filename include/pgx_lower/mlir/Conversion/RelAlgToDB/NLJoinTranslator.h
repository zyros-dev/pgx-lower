#ifndef MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#include "JoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include <mlir/Dialect/DB/IR/DBOps.h>

namespace pgx::mlir::relalg {
class NLJoinTranslator : public JoinTranslator {
   ::mlir::Value vector;
   OrderedAttributes orderedAttributesLeft;
   ::mlir::TupleType tupleType;

   protected:
   ::mlir::Location loc;

   public:
   NLJoinTranslator(std::shared_ptr<JoinImpl> impl) : JoinTranslator(impl), loc(joinOp.getLoc()) {}

   virtual void setInfo(Translator* consumer, ColumnSet requiredAttributes) override;

   void build(::mlir::OpBuilder& builder, TranslatorContext& context);
   virtual void scanHT(TranslatorContext& context, ::mlir::OpBuilder& builder) override;

   void probe(::mlir::OpBuilder& builder, TranslatorContext& context);
   virtual void consume(Translator* child, ::mlir::OpBuilder& builder, TranslatorContext& context) override;
   virtual void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override;
};
} // end namespace pgx::mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
