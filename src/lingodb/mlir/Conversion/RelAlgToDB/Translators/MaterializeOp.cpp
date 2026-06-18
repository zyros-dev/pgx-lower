#include "lingodb/mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "pgx-lower/utility/logging.h"

namespace {

static bool moduleRequestsRowLowerPath(mlir::Operation* op) {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return false;
    }
    auto attr = module->getAttrOfType<mlir::StringAttr>("pgx_lower.lower_path");
    return attr && attr.getValue() == "row";
}

static mlir::StringAttr outputNameFor(mlir::relalg::MaterializeOp materializeOp, size_t index, mlir::OpBuilder& builder) {
    if (index < materializeOp.getColumns().size()) {
        if (auto name = mlir::dyn_cast_or_null<mlir::StringAttr>(materializeOp.getColumns()[index])) {
            return name;
        }
    }
    return builder.getStringAttr("col" + std::to_string(index + 1));
}

static mlir::db::PgRowFieldAttr
rowOutputFieldForValue(mlir::Value value, uint32_t index, mlir::StringAttr outputName, mlir::OpBuilder& builder) {
    mlir::Type type = value.getType();
    if (!mlir::db::isPgValueType(type)) {
        PGX_ERROR("row output materialization requires PostgreSQL semantic value types");
        return {};
    }

    if (auto rowGet = mlir::dyn_cast_or_null<mlir::db::PgRowGetOp>(value.getDefiningOp())) {
        auto inputField = mlir::db::getPgRowFieldByIndex(rowGet.getRow().getType(), rowGet.getIndex());
        if (inputField) {
            return mlir::db::PgRowFieldAttr::get(
                builder.getContext(), index, inputField.getRelid(), inputField.getVarno(), inputField.getAttno(),
                outputName, type, mlir::db::getPgTypeOid(type), mlir::db::getPgTypmod(type),
                mlir::db::getPgCollation(type), mlir::db::getPgNullability(type), false, inputField.getOrigin());
        }
    }

    return mlir::db::PgRowFieldAttr::get(builder.getContext(), index, InvalidOid, 0, 0, outputName, type,
                                         mlir::db::getPgTypeOid(type), mlir::db::getPgTypmod(type),
                                         mlir::db::getPgCollation(type), mlir::db::getPgNullability(type), false,
                                         mlir::db::PgRowFieldOrigin::computed);
}

} // namespace

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
         return "decimal[" + std::to_string(decimalType.getP()) + "," + std::to_string(decimalType.getS()) + "]";
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
      PGX_LOG(RELALG_LOWER, DEBUG, "MaterializeOp::consume called");

      if (context.currentPgRow) {
          std::vector<mlir::Value> values;
          std::vector<mlir::db::PgRowFieldAttr> fields;
          values.reserve(orderedAttributes.getAttrs().size());
          fields.reserve(orderedAttributes.getAttrs().size());

          for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
              auto value = orderedAttributes.resolve(context, i);
              if (!value) {
                  PGX_ERROR("MaterializeOp row output: Column resolution failed for position %zu", i);
                  continue;
              }
              auto field = rowOutputFieldForValue(value, static_cast<uint32_t>(values.size()),
                                                  outputNameFor(materializeOp, i, builder), builder);
              if (!field) {
                  PGX_ERROR("MaterializeOp row output: failed to build output field metadata for position %zu", i);
                  continue;
              }
              values.push_back(value);
              fields.push_back(field);
          }

          auto schema = mlir::db::PgRowSchemaAttr::get(builder.getContext(), fields);
          builder.create<mlir::db::PgEmitRowOp>(materializeOp->getLoc(), values, schema);
          return;
      }

      if (materializeOp.getCols().empty()) {
         builder.create<mlir::dsa::NextRow>(materializeOp->getLoc(), tableBuilder);
         return;
      }
      
      PGX_LOG(RELALG_LOWER, DEBUG, "MaterializeOp: Resolving %zu columns", orderedAttributes.getAttrs().size());
      for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
         auto val = orderedAttributes.resolve(context, i);
         
         if (!val) {
            PGX_ERROR("MaterializeOp: Column resolution failed for position %zu", i);
            continue;
         }
         
         ::mlir::Value valid;
         if (isa<mlir::db::NullableType>(val.getType())) {
            ::mlir::Value isNullResult = builder.create<mlir::db::IsNullOp>(materializeOp->getLoc(), val);
            valid = builder.create<mlir::db::NotOp>(materializeOp->getLoc(), isNullResult);
            val = builder.create<mlir::db::NullableGetVal>(materializeOp->getLoc(), getBaseType(val.getType()), val);
         } else {
            valid = builder.create<mlir::arith::ConstantIntOp>(materializeOp->getLoc(), 1, 1);
         }
         builder.create<mlir::dsa::Append>(materializeOp->getLoc(), tableBuilder, val, valid);
      }
      builder.create<mlir::dsa::NextRow>(materializeOp->getLoc(), tableBuilder);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      PGX_LOG(RELALG_LOWER, DEBUG, "MaterializeOp::produce called");
      const bool rowPath = moduleRequestsRowLowerPath(materializeOp);
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
      auto tupleType = rowPath ? mlir::TupleType::get(builder.getContext(), {})
                               : orderedAttributes.getTupleType(builder.getContext());
      PGX_LOG(RELALG_LOWER, DEBUG, "MaterializeOp: Building description for %zu columns", materializeOp.getColumns().size());
      for (size_t i = 0; !rowPath && i < materializeOp.getColumns().size(); i++) {
          if (!descr.empty()) {
              descr += ";";
          }
          auto colAttr = materializeOp.getColumns()[i];
          if (!colAttr) {
              PGX_ERROR("MaterializeTranslator::produce column attribute at index %zu is null", i);
              continue;
          }

          if (!isa<::mlir::StringAttr>(colAttr)) {
              PGX_ERROR("MaterializeTranslator::produce column attribute is not a StringAttr");
              continue;
          }

          auto colName = cast<::mlir::StringAttr>(colAttr).str();
          auto typeDescr = arrowDescrFromType(getBaseType(tupleType.getType(i)));
          PGX_LOG(RELALG_LOWER, DEBUG, "MaterializeOp: Column %zu: '%s' type '%s'", i, colName.c_str(),
                  typeDescr.c_str());
          descr += colName + ":" + typeDescr;
      }
      PGX_LOG(RELALG_LOWER, DEBUG, "MaterializeOp: Final description string: '%s'", descr.c_str());
      
      tableBuilder = builder.create<mlir::dsa::CreateDS>(materializeOp.getLoc(), mlir::dsa::TableBuilderType::get(builder.getContext(), orderedAttributes.getTupleType(builder.getContext())), builder.getStringAttr(descr));
      
      if (children.empty()) {
         PGX_ERROR("MaterializeTranslator::produce no children!");
         return;
      }
      
      PGX_LOG(RELALG_LOWER, DEBUG, "MaterializeOp: Calling child[0]->produce");
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
