#ifndef MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#include "Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include <mlir/Dialect/RelAlg/IR/Column.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h>

namespace mlir::relalg {

class JoinTranslator;
struct JoinImpl {
   virtual mlir::Value getFlag() { return stopOnFlag ? matchFoundFlag : Value(); }
   virtual void addAdditionalRequiredColumns() {}
   virtual void handleLookup(Value matched, Value markerBefore, pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void afterLookup(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void handleScanned(Value marker, pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void after(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) {}

   mlir::Value matchFoundFlag;
   bool stopOnFlag = true;
   JoinTranslator* translator;
   mlir::Location loc;
   Operator joinOp;
   Value builderChild, lookupChild;
   bool markable;
   JoinImpl(Operator joinOp, Value builderChild, Value lookupChild, bool markable = false) : loc(joinOp->getLoc()), joinOp(joinOp), builderChild(builderChild), lookupChild(lookupChild), markable(markable) {
   }
};
class JoinTranslator : public pgx::mlir::relalg::Translator {
   protected:
   Operator joinOp;
   pgx::mlir::relalg::Translator* builderChild;
   pgx::mlir::relalg::Translator* lookupChild;
   std::shared_ptr<JoinImpl> impl;

   public:
   JoinTranslator(std::shared_ptr<JoinImpl> joinImpl);
   void addJoinRequiredColumns();
   void handleMappingNull(::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context, pgx::mlir::relalg::TranslatorContext::AttributeResolverScope& scope);
   void handleMapping(::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context, pgx::mlir::relalg::TranslatorContext::AttributeResolverScope& scope);
   void handlePotentialMatch(::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context, Value matches, ::mlir::function_ref<void(::mlir::OpBuilder&, pgx::mlir::relalg::TranslatorContext& context, pgx::mlir::relalg::TranslatorContext::AttributeResolverScope&)> onMatch = nullptr);

   virtual void scanHT(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) = 0;
   void forwardConsume(::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) {
      consumer->consume(this, builder, context);
   }

   virtual Value evaluatePredicate(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext::AttributeResolverScope& scope);
   std::vector<size_t> customLookupBuilders;
};
} // end namespace mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
