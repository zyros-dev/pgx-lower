#include "lingodb/mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"

class SortTranslator : public mlir::relalg::Translator {
   mlir::relalg::SortOp sortOp;
   ::mlir::Value vector;
   mlir::relalg::OrderedAttributes orderedAttributes;

   public:
   SortTranslator(mlir::relalg::SortOp sortOp) : mlir::relalg::Translator(sortOp), sortOp(sortOp) {
   }
   virtual void consume(mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      ::mlir::Value packed = orderedAttributes.pack(context, builder, sortOp->getLoc());
      builder.create<mlir::dsa::Append>(sortOp->getLoc(), vector, packed);
   }
   ::mlir::Value createSortPredicate(::mlir::OpBuilder& builder, std::vector<std::pair<::mlir::Value, ::mlir::Value>> sortCriteria, ::mlir::Value trueVal, ::mlir::Value falseVal, size_t pos) {
      if (pos < sortCriteria.size()) {
         ::mlir::Value lt = builder.create<mlir::db::CmpOp>(sortOp->getLoc(), mlir::db::DBCmpPredicate::lt, sortCriteria[pos].first, sortCriteria[pos].second);
         lt = builder.create<mlir::db::DeriveTruth>(sortOp->getLoc(), lt);
         auto ifOp =
             builder.create<mlir::scf::IfOp>(sortOp->getLoc(), mlir::TypeRange{builder.getI1Type()}, lt, true, true);
         {
             auto& thenBlock = ifOp.getThenRegion().front();
             mlir::OpBuilder thenBuilder(&thenBlock, thenBlock.begin());
             thenBuilder.create<mlir::scf::YieldOp>(sortOp->getLoc(), trueVal);
         }
         {
             auto& elseBlock = ifOp.getElseRegion().front();
             mlir::OpBuilder elseBuilder(&elseBlock, elseBlock.begin());
             mlir::Value eq = elseBuilder.create<mlir::db::CmpOp>(sortOp->getLoc(),
                                                                  mlir::db::DBCmpPredicate::eq,
                                                                  sortCriteria[pos].first,
                                                                  sortCriteria[pos].second);
             eq = elseBuilder.create<mlir::db::DeriveTruth>(sortOp->getLoc(), eq);
             auto ifOp2 =
                 elseBuilder.create<mlir::scf::IfOp>(sortOp->getLoc(), mlir::TypeRange{builder.getI1Type()}, eq, true, true);
             {
                 auto& thenBlock2 = ifOp2.getThenRegion().front();
                 mlir::OpBuilder thenBuilder2(&thenBlock2, thenBlock2.begin());
                 thenBuilder2.create<mlir::scf::YieldOp>(
                     sortOp->getLoc(),
                     createSortPredicate(thenBuilder2, sortCriteria, trueVal, falseVal, pos + 1));
             }
             {
                 auto& elseBlock2 = ifOp2.getElseRegion().front();
                 mlir::OpBuilder elseBuilder2(&elseBlock2, elseBlock2.begin());
                 elseBuilder2.create<mlir::scf::YieldOp>(sortOp->getLoc(), falseVal);
             }
             elseBuilder.create<mlir::scf::YieldOp>(sortOp->getLoc(), ifOp2.getResult(0));
         }
         return ifOp.getResult(0);
      } else {
         return falseVal;
      }
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
       auto scope = context.createScope();
       orderedAttributes = mlir::relalg::OrderedAttributes::fromColumns(requiredAttributes);
       const auto tupleType = orderedAttributes.getTupleType(builder.getContext());

       const auto sortSpecs = sortOp.getSortspecs();

       // Get tuple type which has ALL columns
       auto tupleTy = tupleType.cast<mlir::TupleType>();
       int32_t numTotalCols = tupleTy.size();

       // For now, extract type OIDs from sortSpecs for the sort key columns
       // and use placeholder values for non-sort columns
       // TODO: Get actual type OIDs for all columns from SortOp attributes
       std::vector<mlir::Attribute> allTypeOids;
       std::vector<mlir::Attribute> allTypmods;

       // Temporary: assume all columns have same type as first sort key
       // This is WRONG but will compile - needs proper fix in AST translation
       for (int i = 0; i < numTotalCols; i++) {
           if (i < sortSpecs.size()) {
               auto sortSpec = cast<mlir::relalg::SortSpecificationAttr>(sortSpecs[i]);
               allTypeOids.push_back(builder.getI32IntegerAttr(sortSpec.getTypeOid()));
               allTypmods.push_back(builder.getI32IntegerAttr(sortSpec.getTypmod()));
           } else {
               // Use type from first sort key as placeholder
               auto sortSpec = cast<mlir::relalg::SortSpecificationAttr>(sortSpecs[0]);
               allTypeOids.push_back(builder.getI32IntegerAttr(sortSpec.getTypeOid()));
               allTypmods.push_back(builder.getI32IntegerAttr(sortSpec.getTypmod()));
           }
       }

       // Extract sort key information
       std::vector<mlir::Attribute> sortKeyIndices;
       std::vector<mlir::Attribute> sortOpOids;
       std::vector<mlir::Attribute> directions;

       for (size_t i = 0; i < sortSpecs.size(); i++) {
           auto sortSpec = cast<mlir::relalg::SortSpecificationAttr>(sortSpecs[i]);

           sortKeyIndices.push_back(builder.getI32IntegerAttr(i));
           sortOpOids.push_back(builder.getI32IntegerAttr(sortSpec.getSortOpOid()));

           int dir = (sortSpec.getSortSpec() == mlir::relalg::SortSpec::asc) ? 1 : 0;
           directions.push_back(builder.getI32IntegerAttr(dir));
       }

       auto sortStateType = mlir::dsa::SortStateType::get(
           builder.getContext(), tupleType, builder.getArrayAttr(allTypeOids), builder.getArrayAttr(allTypmods),
           builder.getArrayAttr(sortKeyIndices), builder.getArrayAttr(sortOpOids), builder.getArrayAttr(directions));

       vector = builder.create<mlir::dsa::CreateDS>(sortOp.getLoc(), sortStateType);
       children[0]->produce(context, builder);
       builder.create<mlir::dsa::SortOp>(sortOp->getLoc(), vector);
       {
           auto forOp2 = builder.create<mlir::dsa::ForOp>(sortOp->getLoc(), ::mlir::TypeRange{}, vector,
                                                          ::mlir::Value(), ::mlir::ValueRange{});
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