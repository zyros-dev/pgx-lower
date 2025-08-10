#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"

class ProjectionTranslator : public pgx::mlir::relalg::Translator {
   pgx::mlir::relalg::ProjectionOp projectionOp;

   public:
   ProjectionTranslator(pgx::mlir::relalg::ProjectionOp projectionOp) : pgx::mlir::relalg::Translator(projectionOp), projectionOp(projectionOp) {}

   virtual void consume(pgx::mlir::relalg::Translator* child, mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      consumer->consume(this, builder, context);
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~ProjectionTranslator() {}
};

class DistinctProjectionTranslator : public pgx::mlir::relalg::Translator {
   pgx::mlir::relalg::ProjectionOp projectionOp;
   mlir::Value aggrHt;

   pgx::mlir::relalg::OrderedAttributes key;

   mlir::TupleType valTupleType;
   mlir::TupleType entryType;

   public:
   DistinctProjectionTranslator(pgx::mlir::relalg::ProjectionOp projectionOp) : pgx::mlir::relalg::Translator(projectionOp), projectionOp(projectionOp) {
   }

   virtual void consume(pgx::mlir::relalg::Translator* child, mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      mlir::Value emptyVals = builder.create<pgx::mlir::util::UndefOp>(projectionOp->getLoc(), valTupleType);
      mlir::Value packedKey = key.pack(context, builder, projectionOp->getLoc());

      auto reduceOp = builder.create<pgx::mlir::dsa::HashtableInsert>(projectionOp->getLoc(), aggrHt, packedKey, emptyVals);
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      reduceOp.equal().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({packedKey.getType(), packedKey.getType()}, {projectionOp->getLoc(), projectionOp->getLoc()});
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         auto yieldOp = builder.create<pgx::mlir::dsa::YieldOp>(projectionOp->getLoc());
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::Value matches = compareKeys(builder, aggrBuilderBlock->getArgument(0), aggrBuilderBlock->getArgument(1),projectionOp->getLoc());
         builder.create<pgx::mlir::dsa::YieldOp>(projectionOp->getLoc(), matches);
         yieldOp.erase();
      }
      {
         mlir::Block* aggrBuilderBlock = new mlir::Block;
         reduceOp.hash().push_back(aggrBuilderBlock);
         aggrBuilderBlock->addArguments({packedKey.getType()}, {projectionOp->getLoc()});
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::Value hashed = builder.create<pgx::mlir::db::Hash>(projectionOp->getLoc(), builder.getIndexType(), aggrBuilderBlock->getArgument(0));
         builder.create<pgx::mlir::dsa::YieldOp>(projectionOp->getLoc(), hashed);
      }
   }
   mlir::Value compareKeys(mlir::OpBuilder& rewriter, mlir::Value left, mlir::Value right,mlir::Location loc) {
      mlir::Value equal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      auto leftUnpacked = rewriter.create<pgx::mlir::util::UnPackOp>(loc, left);
      auto rightUnpacked = rewriter.create<pgx::mlir::util::UnPackOp>(loc, right);
      for (size_t i = 0; i < leftUnpacked.getNumResults(); i++) {
         mlir::Value compared;
         auto currLeftType = leftUnpacked->getResult(i).getType();
         auto currRightType = rightUnpacked.getResult(i).getType();
         auto currLeftNullableType = currLeftType.dyn_cast_or_null<pgx::mlir::db::NullableType>();
         auto currRightNullableType = currRightType.dyn_cast_or_null<pgx::mlir::db::NullableType>();
         if (currLeftNullableType || currRightNullableType) {
            mlir::Value isNull1 = rewriter.create<pgx::mlir::db::IsNullOp>(loc, rewriter.getI1Type(), leftUnpacked->getResult(i));
            mlir::Value isNull2 = rewriter.create<pgx::mlir::db::IsNullOp>(loc, rewriter.getI1Type(), rightUnpacked->getResult(i));
            mlir::Value anyNull = rewriter.create<mlir::arith::OrIOp>(loc, isNull1, isNull2);
            mlir::Value bothNull = rewriter.create<mlir::arith::AndIOp>(loc, isNull1, isNull2);
            compared = rewriter.create<mlir::scf::IfOp>(
                                  loc, rewriter.getI1Type(), anyNull, [&](mlir::OpBuilder& b, mlir::Location loc) { b.create<mlir::scf::YieldOp>(loc, bothNull); },
                                  [&](mlir::OpBuilder& b, mlir::Location loc) {
                                     mlir::Value left = rewriter.create<pgx::mlir::db::NullableGetVal>(loc, leftUnpacked->getResult(i));
                                     mlir::Value right = rewriter.create<pgx::mlir::db::NullableGetVal>(loc, rightUnpacked->getResult(i));
                                     mlir::Value cmpRes = rewriter.create<pgx::mlir::db::CmpOp>(loc, pgx::mlir::db::DBCmpPredicate::eq, left, right);
                                     b.create<mlir::scf::YieldOp>(loc, cmpRes);
                                  })
                          .getResult(0);
         } else {
            compared = rewriter.create<pgx::mlir::db::CmpOp>(loc, pgx::mlir::db::DBCmpPredicate::eq, leftUnpacked->getResult(i), rightUnpacked.getResult(i));
         }
         mlir::Value localEqual = rewriter.create<mlir::arith::AndIOp>(loc, rewriter.getI1Type(), mlir::ValueRange({equal, compared}));
         equal = localEqual;
      }
      return equal;
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      key = pgx::mlir::relalg::OrderedAttributes::fromRefArr(projectionOp.getCols());
      valTupleType = mlir::TupleType::get(builder.getContext(), {});
      auto keyTupleType = key.getTupleType(builder.getContext());
      mlir::Value emptyTuple = builder.create<pgx::mlir::util::UndefOp>(projectionOp.getLoc(), mlir::TupleType::get(builder.getContext()));
      aggrHt = builder.create<pgx::mlir::dsa::CreateDS>(projectionOp.getLoc(), pgx::mlir::dsa::AggregationHashtableType::get(builder.getContext(), keyTupleType, valTupleType), emptyTuple);

      entryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, valTupleType});
      children[0]->produce(context, builder);

      auto forOp2 = builder.create<pgx::mlir::dsa::ForOp>(projectionOp->getLoc(), mlir::TypeRange{}, aggrHt, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(entryType, projectionOp->getLoc());
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      auto unpacked = builder2.create<pgx::mlir::util::UnPackOp>(projectionOp->getLoc(), forOp2.getInductionVar()).getResults();
      auto unpackedKey = builder2.create<pgx::mlir::util::UnPackOp>(projectionOp->getLoc(), unpacked[0]).getResults();
      key.setValuesForColumns(context, scope, unpackedKey);
      consumer->consume(this, builder2, context);
      builder2.create<pgx::mlir::dsa::YieldOp>(projectionOp->getLoc(), mlir::ValueRange{});

      builder.create<pgx::mlir::dsa::FreeOp>(projectionOp->getLoc(), aggrHt);
   }
   virtual void done() override {
   }
   virtual ~DistinctProjectionTranslator() {}
};
std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::Translator::createProjectionTranslator(pgx::mlir::relalg::ProjectionOp projectionOp) {
   if (projectionOp.set_semantic() == pgx::mlir::relalg::SetSemantic::distinct) {
      return (std::unique_ptr<Translator>) std::make_unique<DistinctProjectionTranslator>(projectionOp);
   } else {
      return (std::unique_ptr<Translator>) std::make_unique<ProjectionTranslator>(projectionOp);
   }
}
