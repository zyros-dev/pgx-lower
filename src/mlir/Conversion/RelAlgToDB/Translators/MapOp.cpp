#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "execution/logging.h"

class MapTranslator : public mlir::relalg::Translator {
   mlir::relalg::MapOp mapOp;

   public:
   MapTranslator(mlir::relalg::MapOp mapOp) : mlir::relalg::Translator(mapOp), mapOp(mapOp) {}

   virtual void consume(mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      auto computedCols = mergeRelationalBlock(
         builder.getInsertionBlock(), op, [](auto x) { return &x->getRegion(0).front(); }, context, scope);
      if (computedCols.size() != mapOp.getComputedCols().size()) {
         PGX_ERROR("MapOp: computed columns size mismatch - expected " + 
                 std::to_string(mapOp.getComputedCols().size()) + 
                 " but got " + std::to_string(computedCols.size()));
         // Cannot continue processing - just return without producing anything
         return;
      }
      for (size_t i = 0; i < computedCols.size(); i++) {
         context.setValueForAttribute(scope, &cast<mlir::relalg::ColumnDefAttr>(mapOp.getComputedCols()[i]).getColumn(), computedCols[i]);
      }
      consumer->consume(this, builder, context);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~MapTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createMapTranslator(mlir::relalg::MapOp mapOp) {
   return std::make_unique<MapTranslator>(mapOp);
}