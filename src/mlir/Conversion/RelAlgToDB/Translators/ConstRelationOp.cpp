#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
class ConstRelTranslator : public pgx::mlir::relalg::Translator {
   pgx::mlir::relalg::ConstRelationOp constRelationOp;

   public:
   ConstRelTranslator(pgx::mlir::relalg::ConstRelationOp constRelationOp) : pgx::mlir::relalg::Translator(constRelationOp), constRelationOp(constRelationOp) {}

   virtual void consume(pgx::mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      assert(false && "should not happen");
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      using namespace mlir;
      pgx::mlir::relalg::OrderedAttributes attributes = pgx::mlir::relalg::OrderedAttributes::fromRefArr(constRelationOp.getColumns());
      auto tupleType = attributes.getTupleType(builder.getContext());
      ::mlir::Value vector = builder.create<pgx::mlir::dsa::CreateDS>(constRelationOp.getLoc(), pgx::mlir::dsa::VectorType::get(builder.getContext(), tupleType));
      for (auto rowAttr : constRelationOp.valuesAttr()) {
         auto row = rowAttr.cast<ArrayAttr>();
         std::vector<Value> values;
         size_t i = 0;
         for (auto entryAttr : row.value()) {
            if (tupleType.getType(i).isa<pgx::mlir::db::NullableType>() && entryAttr.isa<mlir::UnitAttr>()) {
               auto entryVal = builder.create<pgx::mlir::db::NullOp>(constRelationOp->getLoc(), tupleType.getType(i));
               values.push_back(entryVal);
               i++;
            } else {
               ::mlir::Value entryVal = builder.create<pgx::mlir::db::ConstantOp>(constRelationOp->getLoc(), getBaseType(tupleType.getType(i)), entryAttr);
               if (tupleType.getType(i).isa<pgx::mlir::db::NullableType>()) {
                  entryVal = builder.create<pgx::mlir::db::AsNullableOp>(constRelationOp->getLoc(), tupleType.getType(i), entryVal);
               }
               values.push_back(entryVal);
               i++;
            }
         }
         ::mlir::Value packed = builder.create<pgx::mlir::util::PackOp>(constRelationOp->getLoc(), values);
         builder.create<pgx::mlir::dsa::Append>(constRelationOp->getLoc(), vector, packed);
      }
      {
         auto forOp2 = builder.create<pgx::mlir::dsa::ForOp>(constRelationOp->getLoc(), ::mlir::TypeRange{}, vector, ::mlir::Value(), ::mlir::ValueRange{});
         ::mlir::Block* block2 = new ::mlir::Block;
         block2->addArgument(tupleType, constRelationOp->getLoc());
         forOp2.getBodyRegion().push_back(block2);
         ::mlir::OpBuilder builder2(forOp2.getBodyRegion());
         auto unpacked = builder2.create<pgx::mlir::util::UnPackOp>(constRelationOp->getLoc(), forOp2.getInductionVar());
         attributes.setValuesForColumns(context, scope, unpacked.getResults());
         consumer->consume(this, builder2, context);
         builder2.create<pgx::mlir::dsa::YieldOp>(constRelationOp->getLoc(), ::mlir::ValueRange{});
      }
      builder.create<pgx::mlir::dsa::FreeOp>(constRelationOp->getLoc(), vector);
   }
   virtual ~ConstRelTranslator() {}
};

std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::Translator::createConstRelTranslator(pgx::mlir::relalg::ConstRelationOp constRelationOp) {
   return std::make_unique<ConstRelTranslator>(constRelationOp);
}