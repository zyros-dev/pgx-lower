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
      std::vector<::mlir::Value> values;
      std::vector<::mlir::Type> types;
      for (const auto* attr : orderedAttributes.getAttrs()) {
         ::mlir::Value val = context.getValueForAttribute(attr);
         values.push_back(val);
         types.push_back(val.getType());
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

      std::vector<::mlir::Type> columnTypes;
      for (const auto* attr : orderedAttributes.getAttrs()) {
         columnTypes.push_back(attr->type);
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), columnTypes);

      auto& columnManager = builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
      std::vector<mlir::Attribute> sortKeyIndices;

      for (auto specAttr : sortOp.getSortspecs()) {
         auto sortSpec = specAttr.cast<mlir::relalg::SortSpecificationAttr>();
         auto colRef = sortSpec.getAttr();

         for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
            if (orderedAttributes.getAttrs()[i] == &colRef.getColumn()) {
               auto indexAttr = builder.getI32IntegerAttr(static_cast<int32_t>(i));
               auto directionAttr = builder.getI32IntegerAttr(sortSpec.getSortSpec() == mlir::relalg::SortSpec::asc ? 0 : 1);
               sortKeyIndices.push_back(builder.getArrayAttr({indexAttr, directionAttr}));
               break;
            }
         }
      }

      auto sortKeysAttr = builder.getArrayAttr(sortKeyIndices);
      vector = builder.create<mlir::dsa::CreateDS>(sortOp.getLoc(), mlir::dsa::GenericIterableType::get(builder.getContext(), tupleType, "pgsort_iterator"), mlir::Value(), sortKeysAttr);
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