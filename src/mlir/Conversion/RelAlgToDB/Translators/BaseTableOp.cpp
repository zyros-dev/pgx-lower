#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "execution/logging.h"

namespace pgx::mlir::relalg {

class BaseTableTranslator : public Translator {
   static bool registered;
   BaseTableOp baseTableOp;

   public:
   BaseTableTranslator(BaseTableOp baseTableOp) : Translator(baseTableOp), baseTableOp(baseTableOp) {}
   virtual void consume(Translator* child, ::mlir::OpBuilder& builder, TranslatorContext& context) override {
      assert(false && "should not happen");
   }
   
   virtual ColumnSet getAvailableColumns() override {
      ColumnSet available;
      for (auto namedAttr : baseTableOp.getColumnsAttr().getValue()) {
         auto attr = namedAttr.getValue();
         auto attrDef = attr.dyn_cast_or_null<pgx::mlir::relalg::ColumnDefAttr>();
         if (attrDef) {
            available.insert(&attrDef.getColumn());
         }
      }
      return available;
   }
   virtual void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      using namespace mlir;
      std::vector<::mlir::Type> types;
      std::vector<const Column*> cols;
      std::vector<mlir::Attribute> columnNames;
      std::string tableName = baseTableOp->getAttr("table_identifier").cast<mlir::StringAttr>().str();
      std::string scanDescription = R"({ "table": ")" + tableName + R"(", "columns": [ )";
      bool first = true;
      for (auto namedAttr : baseTableOp.getColumnsAttr().getValue()) {
         auto identifier = namedAttr.getName();
         auto attr = namedAttr.getValue();
         auto attrDef = attr.dyn_cast_or_null<pgx::mlir::relalg::ColumnDefAttr>();
         if (requiredAttributes.contains(&attrDef.getColumn())) {
            if (!first) {
               scanDescription += ",";
            } else {
               first = false;
            }
            scanDescription += "\"" + identifier.str() + "\"";
            columnNames.push_back(builder.getStringAttr(identifier.strref()));
            types.push_back(getBaseType(attrDef.getColumn().type));
            cols.push_back(&attrDef.getColumn());
         }
      }
      scanDescription += "] }";

      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      auto recordBatch = pgx::mlir::dsa::RecordBatchType::get(builder.getContext(), tupleType);
      ::mlir::Type chunkIterable = pgx::mlir::dsa::GenericIterableType::get(builder.getContext(), recordBatch, "table_chunk_iterator");

      auto chunkIterator = builder.create<pgx::mlir::dsa::ScanSource>(baseTableOp->getLoc(), chunkIterable, builder.getStringAttr(scanDescription));

      auto forOp = builder.create<pgx::mlir::dsa::ForOp>(baseTableOp->getLoc(), ::mlir::TypeRange{}, chunkIterator, ::mlir::Value(), ::mlir::ValueRange{});
      ::mlir::Block* block = new ::mlir::Block;
      block->addArgument(recordBatch, baseTableOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      ::mlir::OpBuilder builder1(forOp.getBodyRegion());
      auto forOp2 = builder1.create<pgx::mlir::dsa::ForOp>(baseTableOp->getLoc(), ::mlir::TypeRange{}, forOp.getInductionVar(), ::mlir::Value(), ::mlir::ValueRange{});
      ::mlir::Block* block2 = new ::mlir::Block;
      block2->addArgument(recordBatch.getElementType(), baseTableOp->getLoc());
      forOp2.getBodyRegion().push_back(block2);
      ::mlir::OpBuilder builder2(forOp2.getBodyRegion());
      size_t i = 0;
      for (const auto* attr : cols) {
         std::vector<::mlir::Type> types;
         types.push_back(getBaseType(attr->type));
         if (attr->type.isa<pgx::mlir::db::NullableType>()) {
            types.push_back(builder.getI1Type());
         }
         auto atOp = builder2.create<pgx::mlir::dsa::At>(baseTableOp->getLoc(), types, forOp2.getInductionVar(), i++);
         if (attr->type.isa<pgx::mlir::db::NullableType>()) {
            ::mlir::Value isNull = builder2.create<pgx::mlir::db::NotOp>(baseTableOp->getLoc(), atOp.getResult(1));
            ::mlir::Value val = builder2.create<pgx::mlir::db::AsNullableOp>(baseTableOp->getLoc(), attr->type, atOp.getResult(0), isNull);
            context.setValueForAttribute(scope, attr, val);
         } else {
            context.setValueForAttribute(scope, attr, atOp.getResult(0));
         }
      }
      consumer->consume(this, builder2, context);
      builder2.create<pgx::mlir::dsa::YieldOp>(baseTableOp->getLoc());
      builder1.create<pgx::mlir::dsa::YieldOp>(baseTableOp->getLoc());
   }
   virtual ~BaseTableTranslator() {}
};

} // namespace pgx::mlir::relalg

std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::createBaseTableTranslator(::mlir::Operation* op) {
   auto baseTableOp = ::mlir::cast<pgx::mlir::relalg::BaseTableOp>(op);
   return std::make_unique<BaseTableTranslator>(baseTableOp);
}