#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "pgx-lower/utility/logging.h"

class BaseTableTranslator : public mlir::relalg::Translator {
   static bool registered;
   mlir::relalg::BaseTableOp baseTableOp;

   public:
   BaseTableTranslator(mlir::relalg::BaseTableOp baseTableOp) : mlir::relalg::Translator(baseTableOp), baseTableOp(baseTableOp) {
   }
   virtual void consume(mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      PGX_ERROR("BaseTableTranslator::consume called - this should not happen for leaf nodes");
      return;
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      PGX_INFO("BaseTableOp::produce called - registering columns");
      auto scope = context.createScope();
      using namespace mlir;
      std::vector<::mlir::Type> types;
      std::vector<const mlir::relalg::Column*> cols;
      std::vector<::mlir::Attribute> columnNames;
      std::string tableName = cast<::mlir::StringAttr>(baseTableOp->getAttr("table_identifier")).str();
      std::string scanDescription = R"({ "table": ")" + tableName + R"(", "columns": [ )";
      bool first = true;
      
      auto columnsAttr = baseTableOp.getColumnsAttr();
      auto columnOrderAttr = baseTableOp->getAttrOfType<mlir::ArrayAttr>("column_order");
      
      if (!columnsAttr || columnsAttr.empty()) {
         scanDescription += R"("dummy_col")";
         types.push_back(builder.getI32Type());
      } else {
         // If we have column_order, use that order; otherwise fall back to dictionary iteration
         if (columnOrderAttr && !columnOrderAttr.empty()) {
            // Use the specified column order
            for (auto columnNameAttr : columnOrderAttr) {
               auto columnName = columnNameAttr.cast<mlir::StringAttr>().getValue();
               auto namedAttr = columnsAttr.getNamed(columnName);
               if (namedAttr) {
                  auto attrDef = namedAttr->getValue().dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
                  if (requiredAttributes.empty() || requiredAttributes.contains(&attrDef.getColumn())) {
                     if (!first) {
                        scanDescription += ",";
                     } else {
                        first = false;
                     }
                     scanDescription += "\"" + columnName.str() + "\"";
                     columnNames.push_back(builder.getStringAttr(columnName));
                     types.push_back(getBaseType(attrDef.getColumn().type));
                     cols.push_back(&attrDef.getColumn());
                  }
               }
            }
         } else {
            // Fall back to dictionary iteration (alphabetical order)
            for (auto namedAttr : columnsAttr) {
               auto identifier = namedAttr.getName();
               auto attr = namedAttr.getValue();
               auto attrDef = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
               if (requiredAttributes.empty() || requiredAttributes.contains(&attrDef.getColumn())) {
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
         }
      }
      scanDescription += "] }";

      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      auto recordBatch = mlir::dsa::RecordBatchType::get(builder.getContext(), tupleType);
      mlir::Type chunkIterable = mlir::dsa::GenericIterableType::get(builder.getContext(), recordBatch, "table_chunk_iterator");

      auto chunkIterator = builder.create<mlir::dsa::ScanSource>(baseTableOp->getLoc(), chunkIterable, builder.getStringAttr(scanDescription));

      auto forOp = builder.create<mlir::dsa::ForOp>(baseTableOp->getLoc(), mlir::TypeRange{}, chunkIterator, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(recordBatch, baseTableOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      mlir::OpBuilder builder1(forOp.getBodyRegion());
      auto forOp2 = builder1.create<mlir::dsa::ForOp>(baseTableOp->getLoc(), mlir::TypeRange{}, forOp.getInductionVar(), mlir::Value(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(recordBatch.getElementType(), baseTableOp->getLoc());
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      size_t i = 0;
      
      if (cols.empty()) {
         auto atOp = builder2.create<mlir::dsa::At>(baseTableOp->getLoc(), 
                                                    builder.getI32Type(), 
                                                    forOp2.getInductionVar(), 0);
      } else {
         for (const auto* attr : cols) {
            std::vector<::mlir::Type> types;
            types.push_back(getBaseType(attr->type));
            if (isa<mlir::db::NullableType>(attr->type)) {
               types.push_back(builder.getI1Type());
            }
            auto atOp = builder2.create<mlir::dsa::At>(baseTableOp->getLoc(), types, forOp2.getInductionVar(), i++);
            if (isa<mlir::db::NullableType>(attr->type)) {
               PGX_DEBUG("BaseTableOp: Processing nullable column index " + std::to_string(i-1) +
                        " - atOp.getValid() will be inverted with NotOp");
               
               ::mlir::Value isNull = builder2.create<mlir::db::NotOp>(baseTableOp->getLoc(), atOp.getValid());
               ::mlir::Value val = builder2.create<mlir::db::AsNullableOp>(baseTableOp->getLoc(), attr->type, atOp.getVal(), isNull);
               context.setValueForAttribute(scope, attr, val);
            } else {
               PGX_DEBUG("BaseTableOp: Registering non-nullable column " + std::to_string(i-1) + " in scope");
               context.setValueForAttribute(scope, attr, atOp.getVal());
            }
         }
      }
      
      // Only call consume if we have a consumer (MaterializeOp sets this)
      if (consumer) {
         consumer->consume(this, builder2, context);
      }
      builder2.create<mlir::dsa::YieldOp>(baseTableOp->getLoc());
      builder1.create<mlir::dsa::YieldOp>(baseTableOp->getLoc());
   }
   virtual ~BaseTableTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createBaseTableTranslator(mlir::relalg::BaseTableOp baseTableOp) {
   return std::make_unique<BaseTableTranslator>(baseTableOp);
}