#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H

#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

namespace pgx::mlir::relalg {

class JoinImpl {
   public:
   BinaryOperator joinOp;
   Translator* builderChild;
   Translator* lookupChild;
   Translator* translator;
   
   JoinImpl(BinaryOperator joinOp) : joinOp(joinOp) {}
   
   virtual void after(TranslatorContext& context, mlir::OpBuilder& builder) = 0;
   virtual void build(mlir::OpBuilder& builder, TranslatorContext& context) = 0;
   virtual void probe(mlir::OpBuilder& builder, TranslatorContext& context) = 0;
   virtual void addAdditionalRequiredColumns() = 0;
   
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
   
   virtual void scanHT(TranslatorContext& context, mlir::OpBuilder& builder) = 0;
   
   virtual ~JoinTranslator() = default;
};

} // namespace pgx::mlir::relalg

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_JOINTRANSLATOR_H