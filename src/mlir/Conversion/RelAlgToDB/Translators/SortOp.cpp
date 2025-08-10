#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "execution/logging.h"

namespace pgx::mlir::relalg {

class SortTranslator : public Translator {
   SortOp sortOp;
   ::mlir::Value vector;
   OrderedAttributes orderedAttributes;

   public:
   SortTranslator(SortOp sortOp) : Translator(sortOp), sortOp(sortOp) {
   }
   
   virtual ColumnSet getAvailableColumns() override {
      // Sort doesn't change available columns - pass through from child
      if (!children.empty()) {
         return children[0]->getAvailableColumns();
      }
      return ColumnSet();
   }
   virtual void consume(Translator* child, ::mlir::OpBuilder& builder, TranslatorContext& context) override {
      // Collect values from context based on ordered attributes
      std::vector<::mlir::Value> values;
      for (size_t i = 0; i < orderedAttributes.size(); i++) {
         values.push_back(orderedAttributes.resolve(context, i));
      }
      ::mlir::Value packed = builder.create<util::PackOp>(sortOp->getLoc(), values);
      builder.create<dsa::Append>(sortOp->getLoc(), vector, packed);
   }
   ::mlir::Value createSortPredicate(::mlir::OpBuilder& builder, std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria, ::mlir::Value trueVal, ::mlir::Value falseVal, size_t pos) {
      if (pos < sortCriteria.size()) {
         ::mlir::Value lt = builder.create<db::CmpOp>(sortOp->getLoc(), db::DBCmpPredicate::lt, sortCriteria[pos].first, sortCriteria[pos].second);
         lt = builder.create<db::DeriveTruth>(sortOp->getLoc(), lt);
         auto ifOp = builder.create<::mlir::scf::IfOp>(
            sortOp->getLoc(), builder.getI1Type(), lt, [&](::mlir::OpBuilder& builder, ::mlir::Location loc) { builder.create<::mlir::scf::YieldOp>(loc, trueVal); }, [&](::mlir::OpBuilder& builder, ::mlir::Location loc) {
               ::mlir::Value eq = builder.create<db::CmpOp>(loc, db::DBCmpPredicate::eq, sortCriteria[pos].first, sortCriteria[pos].second);
               eq=builder.create<db::DeriveTruth>(sortOp->getLoc(),eq);
               auto ifOp2 = builder.create<::mlir::scf::IfOp>(loc, builder.getI1Type(), eq,[&](::mlir::OpBuilder& builder, ::mlir::Location loc) {
                  builder.create<::mlir::scf::YieldOp>(loc, createSortPredicate(builder, sortCriteria, trueVal, falseVal, pos + 1));
                  },[&](::mlir::OpBuilder& builder, ::mlir::Location loc) {
                     builder.create<::mlir::scf::YieldOp>(loc, falseVal);
               });
               builder.create<::mlir::scf::YieldOp>(loc, ifOp2.getResult(0)); });
         return ifOp.getResult(0);
      } else {
         return falseVal;
      }
   }
   virtual void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      orderedAttributes = OrderedAttributes::fromColumns(requiredAttributes);
      auto tupleType = orderedAttributes.getTupleType(builder.getContext());
      vector = builder.create<dsa::CreateDS>(sortOp.getLoc(), dsa::VectorType::get(builder.getContext(), tupleType));
      children[0]->produce(context,builder);
      {
         auto dbSortOp = builder.create<dsa::SortOp>(sortOp->getLoc(), vector);
         ::mlir::Block* block2 = new ::mlir::Block;
         block2->addArgument(tupleType, sortOp->getLoc());
         block2->addArguments(tupleType, sortOp->getLoc());
         dbSortOp.region().push_back(block2);
         ::mlir::OpBuilder builder2(dbSortOp.region());
         auto unpackedLeft = builder2.create<util::UnPackOp>(sortOp->getLoc(), block2->getArgument(0));
         auto unpackedRight = builder2.create<util::UnPackOp>(sortOp->getLoc(), block2->getArgument(1));
         std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
         for (auto attr : sortOp.sortspecs()) {
            auto sortspecAttr = attr.cast<SortSpecificationAttr>();
            auto pos = orderedAttributes.getPos(&sortspecAttr.getAttr().getColumn());
            ::mlir::Value left = unpackedLeft.getResult(pos);
            ::mlir::Value right = unpackedRight.getResult(pos);
            if (sortspecAttr.getSortSpec() == SortSpec::desc) {
               std::swap(left, right);
            }
            sortCriteria.push_back({left, right});
         }
         auto trueVal = builder2.create<db::ConstantOp>(sortOp->getLoc(), builder.getI1Type(), builder.getIntegerAttr(builder.getI64Type(), 1));
         auto falseVal = builder2.create<db::ConstantOp>(sortOp->getLoc(), builder.getI1Type(), builder.getIntegerAttr(builder.getI64Type(), 0));

         builder2.create<dsa::YieldOp>(sortOp->getLoc(), createSortPredicate(builder2, sortCriteria, trueVal, falseVal, 0));
      }
      {
         auto forOp2 = builder.create<dsa::ForOp>(sortOp->getLoc(), ::mlir::TypeRange{}, vector, mlir::Value(), ::mlir::ValueRange{});
         ::mlir::Block* block2 = new ::mlir::Block;
         block2->addArgument(tupleType, sortOp->getLoc());
         forOp2.getBodyRegion().push_back(block2);
         ::mlir::OpBuilder builder2(forOp2.getBodyRegion());
         auto unpacked = builder2.create<util::UnPackOp>(sortOp->getLoc(), forOp2.getInductionVar());
         orderedAttributes.setValuesForColumns(context, scope, unpacked.getResults());
         consumer->consume(this, builder2, context);
         builder2.create<dsa::YieldOp>(sortOp->getLoc(), ::mlir::ValueRange{});
      }
      builder.create<dsa::FreeOp>(sortOp->getLoc(), vector);
   }

   virtual ~SortTranslator() {}
};

std::unique_ptr<Translator> Translator::createSortTranslator(SortOp sortOp) {
   return std::make_unique<SortTranslator>(sortOp);
}