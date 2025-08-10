#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H

#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

namespace pgx::mlir::relalg {

// Forward declaration
class JoinTranslator;

class JoinImpl {
   public:
   BinaryOperator joinOp;
   ::mlir::Value builderChild;
   ::mlir::Value lookupChild;
   JoinTranslator* translator;
   ::mlir::Location loc;
   bool markable;
   ::mlir::Value matchFoundFlag;
   bool stopOnFlag = true;
   
   JoinImpl(BinaryOperator joinOp, ::mlir::Value builderChild, ::mlir::Value lookupChild, bool markable = false) 
       : joinOp(joinOp), builderChild(builderChild), lookupChild(lookupChild), 
         loc(joinOp->getLoc()), markable(markable) {}
   
   virtual ::mlir::Value getFlag() { return stopOnFlag ? matchFoundFlag : ::mlir::Value(); }
   virtual void addAdditionalRequiredColumns() {}
   virtual void handleLookup(::mlir::Value matched, ::mlir::Value markerBefore, TranslatorContext& context, ::mlir::OpBuilder& builder) = 0;
   virtual void beforeLookup(TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void afterLookup(TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void handleScanned(::mlir::Value marker, TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   virtual void after(TranslatorContext& context, ::mlir::OpBuilder& builder) {}
   
   virtual ~JoinImpl() = default;
};

class JoinTranslator : public Translator {
   protected:
   BinaryOperator joinOp;
   std::shared_ptr<JoinImpl> impl;
   Translator* builderChild;
   Translator* lookupChild;
   
   public:
   JoinTranslator(std::shared_ptr<JoinImpl> joinImpl);
   
   void addJoinRequiredColumns();
   void handleMappingNull(::mlir::OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);
   void handleMapping(::mlir::OpBuilder& builder, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);
   void handlePotentialMatch(::mlir::OpBuilder& builder, TranslatorContext& context, ::mlir::Value matches, 
                           ::mlir::function_ref<void(::mlir::OpBuilder&, TranslatorContext& context, TranslatorContext::AttributeResolverScope&)> onMatch = nullptr);
   
   virtual void scanHT(TranslatorContext& context, ::mlir::OpBuilder& builder) = 0;
   void forwardConsume(::mlir::OpBuilder& builder, TranslatorContext& context) {
      consumer->consume(this, builder, context);
   }
   
   virtual ::mlir::Value evaluatePredicate(TranslatorContext& context, ::mlir::OpBuilder& builder, TranslatorContext::AttributeResolverScope& scope);
   std::vector<size_t> customLookupBuilders;
   
   virtual ColumnSet getAvailableColumns() override;
   
   virtual ~JoinTranslator() = default;
};

} // namespace pgx::mlir::relalg

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H