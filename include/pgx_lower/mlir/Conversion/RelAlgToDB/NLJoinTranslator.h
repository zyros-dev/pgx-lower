#ifndef MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
#include "JoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include <mlir/Dialect/DB/IR/DBOps.h>

namespace pgx::mlir::relalg {
class NLJoinTranslator : public pgx::mlir::relalg::JoinTranslator {
   Value vector;
   pgx::mlir::relalg::OrderedAttributes orderedAttributesLeft;
   mlir::TupleType tupleType;

   protected:
   ::mlir::Location loc;

   public:
   NLJoinTranslator(std::shared_ptr<JoinImpl> impl) : JoinTranslator(impl), loc(joinOp.getLoc()) {}

   virtual void setInfo(pgx::mlir::relalg::Translator* consumer, pgx::mlir::relalg::ColumnSet requiredAttributes) override;

   void build(::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context);
   virtual void scanHT(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override;

   void probe(::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context);
   virtual void consume(pgx::mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override;
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override;
};
} // end namespace pgx::mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_NLJOINTRANSLATOR_H
