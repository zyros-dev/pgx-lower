#include "lingodb/mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "lingodb/runtime/PgSortRuntime.h"
#include <unordered_map>
#include <string>

class SortTranslator : public mlir::relalg::Translator {
   mlir::relalg::SortOp sortOp;
   ::mlir::Value vector;
   mlir::relalg::OrderedAttributes orderedAttributes;

   public:
   SortTranslator(mlir::relalg::SortOp sortOp) : mlir::relalg::Translator(sortOp), sortOp(sortOp) {
   }
   virtual void consume(mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
       // Since things can randomly become null, just force them to be null instead. incredible.
       // TODO: NV, if you feel like saving one byte per typle, you can change this!
      std::vector<::mlir::Value> values;
      std::vector<::mlir::Type> types;
      for (const auto* attr : orderedAttributes.getAttrs()) {
         ::mlir::Value val = context.getValueForAttribute(attr);

         if (!mlir::isa<mlir::db::NullableType>(val.getType())) {
            auto nullableType = mlir::db::NullableType::get(builder.getContext(), val.getType());
            val = builder.create<mlir::db::AsNullableOp>(sortOp->getLoc(), nullableType, val);
            types.push_back(nullableType);
         } else {
            types.push_back(val.getType());
         }
         values.push_back(val);
      }

      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      ::mlir::Value packed = values.empty()
         ? builder.create<mlir::util::UndefOp>(sortOp->getLoc(), tupleType).getResult()
         : builder.create<mlir::util::PackOp>(sortOp->getLoc(), tupleType, values).getResult();

      builder.create<mlir::dsa::Append>(sortOp->getLoc(), vector, packed);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      orderedAttributes = mlir::relalg::OrderedAttributes::fromColumns(requiredAttributes);

      // PGX-LOWER: Reorder SortSpecification to match actual tuple layout from orderedAttributes
      if (auto specPtrAttr = sortOp.getSpecPtrAttr()) {
         auto specPtr = specPtrAttr.cast<mlir::IntegerAttr>().getInt();
         auto* spec = reinterpret_cast<runtime::SortSpecification*>(specPtr);

         // Build map from column pointer to new position in orderedAttributes
         std::unordered_map<std::string, size_t> newPositions;
         auto& columnManager = builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();

         for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
            auto [scope, name] = columnManager.getName(orderedAttributes.getAttrs()[i]);
            std::string key = scope + "." + name;
            newPositions[key] = i;
         }

         // Reorder spec->columns to match orderedAttributes order
         std::vector<runtime::SortColumnInfo> reorderedColumns(spec->num_columns);
         for (int32_t i = 0; i < spec->num_columns; i++) {
            std::string key = std::string(spec->columns[i].table_name) + "." + std::string(spec->columns[i].column_name);
            auto it = newPositions.find(key);
            if (it != newPositions.end()) {
               reorderedColumns[it->second] = spec->columns[i];
            }
         }

         // Write back reordered columns
         for (int32_t i = 0; i < spec->num_columns; i++) {
            spec->columns[i] = reorderedColumns[i];
         }

         // Update sort_key_indices to point to new positions
         for (int32_t i = 0; i < spec->num_sort_keys; i++) {
            int32_t oldIdx = spec->sort_key_indices[i];
            if (oldIdx >= 0 && oldIdx < spec->num_columns) {
               std::string key = std::string(reorderedColumns[oldIdx].table_name) + "." + std::string(reorderedColumns[oldIdx].column_name);
               auto it = newPositions.find(key);
               if (it != newPositions.end()) {
                  spec->sort_key_indices[i] = static_cast<int32_t>(it->second);
               }
            }
         }
      }

      std::vector<::mlir::Type> nullableTypes;
      for (const auto* attr : orderedAttributes.getAttrs()) {
         ::mlir::Type attrType = attr->type;
         if (!mlir::isa<mlir::db::NullableType>(attrType)) {
            attrType = mlir::db::NullableType::get(builder.getContext(), attrType);
         }
         nullableTypes.push_back(attrType);
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), nullableTypes);

      // Use GenericIterableType with "pgsort_iterator" to distinguish from regular Vector
      vector = builder.create<mlir::dsa::CreateDS>(sortOp.getLoc(), mlir::dsa::GenericIterableType::get(builder.getContext(), tupleType, "pgsort_iterator"), sortOp.getSpecPtrAttr());
      children[0]->produce(context,builder);
      builder.create<mlir::dsa::SortOp>(sortOp->getLoc(), vector);
      {
         auto forOp2 = builder.create<mlir::dsa::ForOp>(sortOp->getLoc(), ::mlir::TypeRange{}, vector, ::mlir::Value(), ::mlir::ValueRange{});
         ::mlir::Block* block2 = new ::mlir::Block;
         block2->addArgument(tupleType, sortOp->getLoc());
         forOp2.getBodyRegion().push_back(block2);
         ::mlir::OpBuilder builder2(forOp2.getBodyRegion());
         auto unpacked = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), forOp2.getInductionVar());
         orderedAttributes.setValuesForColumns(context, scope, unpacked.getResults());
         consumer->consume(this, builder2, context);
         builder2.create<mlir::dsa::YieldOp>(sortOp->getLoc(), ::mlir::ValueRange{});
      }
      builder.create<mlir::dsa::FreeOp>(sortOp->getLoc(), vector);
   }

   virtual ~SortTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createSortTranslator(mlir::relalg::SortOp sortOp) {
   return std::make_unique<SortTranslator>(sortOp);
}