#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "core/logging.h"

namespace pgx::mlir::relalg {

class MapTranslator : public Translator {
   pgx::mlir::relalg::MapOp mapOp;

   public:
   MapTranslator(pgx::mlir::relalg::MapOp mapOp) : pgx::mlir::relalg::Translator(mapOp), mapOp(mapOp) {}

   virtual void consume(pgx::mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      auto computedCols = mergeRelationalBlock(
         builder.getInsertionBlock(), op, [](auto x) { return &x->getRegion(0).front(); }, context, scope);
      assert(computedCols.size() == mapOp.getComputedCols().size());
      for (size_t i = 0; i < computedCols.size(); i++) {
         context.setValueForAttribute(scope, &mapOp.getComputedCols()[i].cast<pgx::mlir::relalg::ColumnDefAttr>().getColumn(), computedCols[i]);
      }
      consumer->consume(this, builder, context);
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }
   
   virtual ColumnSet getAvailableColumns() override {
      ColumnSet available;
      // Map adds computed columns to available columns from child
      if (!children.empty()) {
         available = children[0]->getAvailableColumns();
      }
      for (auto computedCol : mapOp.getComputedCols()) {
         available.insert(&computedCol.cast<pgx::mlir::relalg::ColumnDefAttr>().getColumn());
      }
      return available;
   }

   virtual ~MapTranslator() {}
};

} // namespace pgx::mlir::relalg

std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::createMapTranslator(::mlir::Operation* op) {
   auto mapOp = ::mlir::cast<pgx::mlir::relalg::MapOp>(op);
   return std::make_unique<MapTranslator>(mapOp);
}