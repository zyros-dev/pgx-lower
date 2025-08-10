#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"

class MapTranslator : public pgx::mlir::relalg::Translator {
   pgx::mlir::relalg::MapOp mapOp;

   public:
   MapTranslator(pgx::mlir::relalg::MapOp mapOp) : pgx::mlir::relalg::Translator(mapOp), mapOp(mapOp) {}

   virtual void consume(pgx::mlir::relalg::Translator* child, mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      auto computedCols = mergeRelationalBlock(
         builder.getInsertionBlock(), op, [](auto x) { return &x->getRegion(0).front(); }, context, scope);
      assert(computedCols.size() == mapOp.computed_cols().size());
      for (size_t i = 0; i < computedCols.size(); i++) {
         context.setValueForAttribute(scope, &mapOp.computed_cols()[i].cast<pgx::mlir::relalg::ColumnDefAttr>().getColumn(), computedCols[i]);
      }
      consumer->consume(this, builder, context);
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~MapTranslator() {}
};

std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::Translator::createMapTranslator(pgx::mlir::relalg::MapOp mapOp) {
   return std::make_unique<MapTranslator>(mapOp);
}