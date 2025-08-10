#ifndef MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#include "Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include <mlir/Dialect/RelAlg/IR/Column.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h>

namespace pgx::mlir::relalg {

class JoinTranslator;
struct JoinImpl {
   virtual ::mlir::Value getFlag() { return stopOnFlag ? matchFoundFlag : ::mlir::Value(); }
   virtual void addAdditionalRequiredColumns() {}
   virtual void handleLookup(::mlir::Value matched, ::mlir::Value markerBefore, TranslatorContext& context, ::mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void afterLookup(TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void handleScanned(::mlir::Value marker, TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void after(TranslatorContext& context, ::mlir::OpBuilder& builder) {}

   ::mlir::Value matchFoundFlag;
   bool stopOnFlag = true;
   JoinTranslator* translator;
   ::mlir::Location loc;
   Operator joinOp;
   ::mlir::Value builderChild, lookupChild;
   bool markable;
   JoinImpl(Operator joinOp, ::mlir::Value builderChild, ::mlir::Value lookupChild, bool markable = false) : loc(joinOp->getLoc()), joinOp(joinOp), builderChild(builderChild), lookupChild(lookupChild), markable(markable) {
   }
};
class JoinTranslator : public Translator {
   protected:
   Operator joinOp;
   Translator* builderChild;
   Translator* lookupChild;
   std::shared_ptr<JoinImpl> impl;

   public:
   JoinTranslator(std::shared_ptr<JoinImpl> joinImpl);
   void addJoinRequiredColumns();
   void handleMappingNull(::mlir::OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);
   void handleMapping(::mlir::OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);
   void handlePotentialMatch(::mlir::OpBuilder& builder, TranslatorContext& context, ::mlir::Value matches, ::mlir::function_ref<void(::mlir::OpBuilder&, TranslatorContext& context, TranslatorContext::AttributeResolverScope&)> onMatch = nullptr);

   virtual void scanHT(TranslatorContext& context, ::mlir::OpBuilder& builder) = 0;
   void forwardConsume(::mlir::OpBuilder& builder, TranslatorContext& context) {
      consumer->consume(this, builder, context);
   }

   virtual ::mlir::Value evaluatePredicate(TranslatorContext& context, ::mlir::OpBuilder& builder, TranslatorContext::AttributeResolverScope& scope);
   virtual ColumnSet getAvailableColumns() override;
   std::vector<size_t> customLookupBuilders;
};
} // end namespace pgx::mlir::relalg
#endif // MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
