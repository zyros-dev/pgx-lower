#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "pgx-lower/utility/logging.h"

class MapTranslator : public mlir::relalg::Translator {
   mlir::relalg::MapOp mapOp;

   public:
   MapTranslator(mlir::relalg::MapOp mapOp) : mlir::relalg::Translator(mapOp), mapOp(mapOp) {}

   virtual void consume(mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      PGX_INFO("MapOp::consume called - processing map operation");
      auto scope = context.createScope();
      PGX_INFO("MapOp: Merging relational block to compute map expressions");
      auto computedCols = mergeRelationalBlock(
         builder.getInsertionBlock(), op, [](auto x) { return &x->getRegion(0).front(); }, context, scope);
      if (computedCols.size() != mapOp.getComputedCols().size()) {
         PGX_ERROR("MapOp: computed columns size mismatch - expected " + 
                 std::to_string(mapOp.getComputedCols().size()) + 
                 " but got " + std::to_string(computedCols.size()));
         return;
      }
      PGX_INFO("MapOp: Registering " + std::to_string(computedCols.size()) + " computed columns");
      for (size_t i = 0; i < computedCols.size(); i++) {
         auto& column = cast<mlir::relalg::ColumnDefAttr>(mapOp.getComputedCols()[i]).getColumn();
         context.setValueForAttribute(scope, &column, computedCols[i]);
      }
      
      PGX_INFO("MapOp: Calling consumer->consume");
      consumer->consume(this, builder, context);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      PGX_INFO("MapOp::produce called - calling child produce");
      if (children.empty()) {
         PGX_ERROR("MapOp::produce - no children!");
         return;
      }
      children[0]->produce(context, builder);
   }

   virtual ~MapTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createMapTranslator(mlir::relalg::MapOp mapOp) {
   return std::make_unique<MapTranslator>(mapOp);
}