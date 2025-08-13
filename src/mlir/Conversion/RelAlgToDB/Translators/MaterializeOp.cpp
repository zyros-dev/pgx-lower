#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "execution/logging.h"

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
      PGX_DEBUG("MaterializeTranslator: Constructor called");
      
      // Verify materializeOp is valid
      if (!materializeOp) {
         PGX_ERROR("MaterializeTranslator: materializeOp is null!");
         return;
      }
      
      // Check if we have columns
      auto cols = materializeOp.getCols();
      if (!cols) {
         PGX_ERROR("MaterializeTranslator: materializeOp.getCols() returned null!");
         return;
      }
      
      PGX_DEBUG("MaterializeTranslator: Processing " + std::to_string(cols.size()) + " columns");
      
      // Handle empty columns case gracefully
      if (cols.empty()) {
         PGX_WARNING("MaterializeTranslator: Empty columns array - creating minimal translator");
         // Don't try to create ordered attributes from empty array
      } else {
         // Create ordered attributes
         orderedAttributes = mlir::relalg::OrderedAttributes::fromRefArr(materializeOp.getCols());
      }
      
      PGX_DEBUG("MaterializeTranslator: Constructor completed successfully");
   }
   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) override {
      PGX_DEBUG("MaterializeTranslator::setInfo called");
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      
      // Only process columns if they exist
      auto cols = materializeOp.getCols();
      if (!cols.empty()) {
         this->requiredAttributes.insert(mlir::relalg::ColumnSet::fromArrayAttr(cols));
      }
      
      PGX_DEBUG("MaterializeTranslator::setInfo calling propagateInfo");
      propagateInfo();
      PGX_DEBUG("MaterializeTranslator::setInfo completed");
   }
   virtual mlir::relalg::ColumnSet getAvailableColumns() override {
      return {};
   }
   virtual void consume(mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      PGX_DEBUG("MaterializeTranslator::consume called");
      
      // Handle empty columns case
      if (materializeOp.getCols().empty()) {
         PGX_DEBUG("MaterializeTranslator::consume - Empty columns, just creating NextRow");
         builder.create<mlir::dsa::NextRow>(materializeOp->getLoc(), tableBuilder);
         return;
      }
      
      for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
         auto val = orderedAttributes.resolve(context, i);
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
      PGX_DEBUG("MaterializeTranslator::produce called");
      
      // Handle empty columns case
      if (materializeOp.getCols().empty()) {
         PGX_WARNING("MaterializeTranslator::produce - Empty columns, creating minimal table");
         
         // Create a minimal table with empty tuple type
         auto emptyTupleType = mlir::TupleType::get(builder.getContext(), {});
         auto tableBuilderType = mlir::dsa::TableBuilderType::get(builder.getContext(), emptyTupleType);
         
         tableBuilder = builder.create<mlir::dsa::CreateDS>(
            materializeOp.getLoc(), 
            tableBuilderType, 
            builder.getStringAttr("")  // Empty description
         );
         
         // Still need to process children
         if (!children.empty()) {
            children[0]->produce(context, builder);
         }
         
         table = builder.create<mlir::dsa::Finalize>(
            materializeOp.getLoc(), 
            mlir::dsa::TableType::get(builder.getContext()), 
            tableBuilder
         ).getRes();
         
         PGX_DEBUG("MaterializeTranslator::produce completed (empty columns case)");
         return;
      }
      
      std::string descr = "";
      PGX_DEBUG("MaterializeTranslator::produce getting tuple type");
      auto tupleType = orderedAttributes.getTupleType(builder.getContext());
      
      PGX_DEBUG("MaterializeTranslator::produce building column description string");
      for (size_t i = 0; i < materializeOp.getColumns().size(); i++) {
         if (!descr.empty()) {
            descr += ";";
         }
         auto colAttr = materializeOp.getColumns()[i];
         if (!colAttr) {
            PGX_ERROR("MaterializeTranslator::produce column attribute at index " + std::to_string(i) + " is null");
            continue;
         }
         
         // Safety check for cast
         if (!isa<::mlir::StringAttr>(colAttr)) {
            PGX_ERROR("MaterializeTranslator::produce column attribute is not a StringAttr");
            continue;
         }
         
         descr += cast<::mlir::StringAttr>(colAttr).str() + ":" + arrowDescrFromType(getBaseType(tupleType.getType(i)));
      }
      
      PGX_DEBUG("MaterializeTranslator::produce column description: " + descr);
      
      PGX_DEBUG("MaterializeTranslator::produce creating CreateDS operation");
      tableBuilder = builder.create<mlir::dsa::CreateDS>(materializeOp.getLoc(), mlir::dsa::TableBuilderType::get(builder.getContext(), orderedAttributes.getTupleType(builder.getContext())), builder.getStringAttr(descr));
      
      PGX_DEBUG("MaterializeTranslator::produce calling child produce");
      if (children.empty()) {
         PGX_ERROR("MaterializeTranslator::produce no children!");
         return;
      }
      
      children[0]->produce(context, builder);
      
      PGX_DEBUG("MaterializeTranslator::produce creating Finalize operation");
      table = builder.create<mlir::dsa::Finalize>(materializeOp.getLoc(), mlir::dsa::TableType::get(builder.getContext()), tableBuilder).getRes();
      
      PGX_DEBUG("MaterializeTranslator::produce completed");
   }
   virtual void done() override {
      materializeOp.replaceAllUsesWith(table);
   }
   virtual ~MaterializeTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createMaterializeTranslator(mlir::relalg::MaterializeOp materializeOp) {
   PGX_DEBUG("createMaterializeTranslator called");
   
   if (!materializeOp) {
      PGX_ERROR("createMaterializeTranslator: materializeOp is null!");
      return nullptr;
   }
   
   PGX_DEBUG("createMaterializeTranslator: Creating MaterializeTranslator instance");
   return std::make_unique<MaterializeTranslator>(materializeOp);
}
