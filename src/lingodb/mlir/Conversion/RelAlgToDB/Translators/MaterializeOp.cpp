#include "lingodb/mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "pgx-lower/execution/logging.h"

class MaterializeTranslator : public mlir::relalg::Translator {
   mlir::relalg::MaterializeOp materializeOp;
   ::mlir::Value tableBuilder;
   ::mlir::Value table;
   mlir::relalg::OrderedAttributes orderedAttributes;
   std::string arrowDescrFromType(::mlir::Type type) {
      if (isIntegerType(type, 1)) {
         return "bool";
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         return "int[" + std::to_string(intWidth) + "]";
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         return "uint[" + std::to_string(uIntWidth) + "]";
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         // TODO: actually handle cases where 128 bits are insufficient.
         auto prec = std::min(decimalType.getP(), 38);
         return "decimal[" + std::to_string(prec) + "," + std::to_string(decimalType.getS()) + "]";
      } else if (auto floatType = type.dyn_cast_or_null<::mlir::FloatType>()) {
         return "float[" + std::to_string(floatType.getWidth()) + "]";
      } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
         return "string";
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
         if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
            return "date[32]";
         } else {
            return "date[64]";
         }
      } else if (auto charType = type.dyn_cast_or_null<mlir::db::CharType>()) {
         return "fixed_sized[" + std::to_string(charType.getBytes()) + "]";
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            return "interval_months";
         } else {
            return "interval_daytime";
         }
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
         return "timestamp[" + std::to_string(static_cast<uint32_t>(timestampType.getUnit())) + "]";
      }
      return "";
   }

   public:
   MaterializeTranslator(mlir::relalg::MaterializeOp materializeOp) : mlir::relalg::Translator(materializeOp.getRel()), materializeOp(materializeOp) {
      if (!materializeOp) {
         PGX_ERROR("MaterializeTranslator: materializeOp is null!");
         return;
      }
      
      auto cols = materializeOp.getCols();
      if (!cols) {
         PGX_ERROR("MaterializeTranslator: materializeOp.getCols() returned null!");
         return;
      }
      
      if (!cols.empty()) {
         orderedAttributes = mlir::relalg::OrderedAttributes::fromRefArr(materializeOp.getCols());
      }
   }
   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      
      auto cols = materializeOp.getCols();
      if (!cols.empty()) {
         this->requiredAttributes.insert(mlir::relalg::ColumnSet::fromArrayAttr(cols));
      }
      
      propagateInfo();
   }
   virtual mlir::relalg::ColumnSet getAvailableColumns() override {
      return {};
   }
   virtual void consume(mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      PGX_INFO("MaterializeOp::consume called");
      
      if (materializeOp.getCols().empty()) {
         builder.create<mlir::dsa::NextRow>(materializeOp->getLoc(), tableBuilder);
         return;
      }
      
      PGX_INFO("MaterializeOp: Resolving " + std::to_string(orderedAttributes.getAttrs().size()) + " columns");
      for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
         auto val = orderedAttributes.resolve(context, i);
         
         if (!val) {
            PGX_ERROR("MaterializeOp: Column resolution failed for position " + std::to_string(i));
            // Skip this column or create a placeholder - for now just skip
            continue;
         }
         
         ::mlir::Value valid;
         if (isa<mlir::db::NullableType>(val.getType())) {
            valid = builder.create<mlir::db::IsNullOp>(materializeOp->getLoc(), val);
            valid = builder.create<mlir::db::NotOp>(materializeOp->getLoc(), valid);
            val = builder.create<mlir::db::NullableGetVal>(materializeOp->getLoc(), getBaseType(val.getType()), val);
         }
         builder.create<mlir::dsa::Append>(materializeOp->getLoc(), tableBuilder, val, valid);
      }
      builder.create<mlir::dsa::NextRow>(materializeOp->getLoc(), tableBuilder);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      PGX_INFO("MaterializeOp::produce called");
      if (materializeOp.getCols().empty()) {
         auto emptyTupleType = mlir::TupleType::get(builder.getContext(), {});
         auto tableBuilderType = mlir::dsa::TableBuilderType::get(builder.getContext(), emptyTupleType);
         
         tableBuilder = builder.create<mlir::dsa::CreateDS>(
            materializeOp.getLoc(), 
            tableBuilderType, 
            builder.getStringAttr("")
         );
         
         if (!children.empty()) {
            children[0]->produce(context, builder);
         }
         
         table = builder.create<mlir::dsa::Finalize>(
            materializeOp.getLoc(), 
            mlir::dsa::TableType::get(builder.getContext()), 
            tableBuilder
         ).getRes();
         
         return;
      }
      
      std::string descr = "";
      auto tupleType = orderedAttributes.getTupleType(builder.getContext());
      for (size_t i = 0; i < materializeOp.getColumns().size(); i++) {
         if (!descr.empty()) {
            descr += ";";
         }
         auto colAttr = materializeOp.getColumns()[i];
         if (!colAttr) {
            PGX_ERROR("MaterializeTranslator::produce column attribute at index " + std::to_string(i) + " is null");
            continue;
         }
         
         if (!isa<::mlir::StringAttr>(colAttr)) {
            PGX_ERROR("MaterializeTranslator::produce column attribute is not a StringAttr");
            continue;
         }
         
         descr += cast<::mlir::StringAttr>(colAttr).str() + ":" + arrowDescrFromType(getBaseType(tupleType.getType(i)));
      }
      
      tableBuilder = builder.create<mlir::dsa::CreateDS>(materializeOp.getLoc(), mlir::dsa::TableBuilderType::get(builder.getContext(), orderedAttributes.getTupleType(builder.getContext())), builder.getStringAttr(descr));
      
      if (children.empty()) {
         PGX_ERROR("MaterializeTranslator::produce no children!");
         return;
      }
      
      PGX_INFO("MaterializeOp: Calling child[0]->produce");
      children[0]->produce(context, builder);
      
      table = builder.create<mlir::dsa::Finalize>(materializeOp.getLoc(), mlir::dsa::TableType::get(builder.getContext()), tableBuilder).getRes();
   }
   virtual void done() override {
      materializeOp.replaceAllUsesWith(table);
   }
   virtual ~MaterializeTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createMaterializeTranslator(mlir::relalg::MaterializeOp materializeOp) {
   if (!materializeOp) {
      PGX_ERROR("createMaterializeTranslator: materializeOp is null!");
      return nullptr;
   }
   
   return std::make_unique<MaterializeTranslator>(materializeOp);
}
