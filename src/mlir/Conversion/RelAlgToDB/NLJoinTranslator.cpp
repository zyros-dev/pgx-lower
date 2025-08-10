#include "mlir/Conversion/RelAlgToDB/NLJoinTranslator.h"
#include <mlir/Dialect/DSA/IR/DSAOps.h>
using namespace pgx::mlir::relalg;
void NLJoinTranslator::setInfo(pgx::mlir::relalg::Translator* consumer, pgx::mlir::relalg::ColumnSet requiredAttributes) {
   this->consumer = consumer;
   this->requiredAttributes = requiredAttributes;
   addJoinRequiredColumns();
   impl->addAdditionalRequiredColumns();
   propagateInfo();
}

void NLJoinTranslator::build(mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) {
   auto const0 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerType(64), builder.getI64IntegerAttr(0));
   mlir::Value packed = orderedAttributesLeft.pack(context, builder, op->getLoc(), impl->markable ? std::vector<Value>{const0} : std::vector<Value>());
   builder.create<pgx::mlir::dsa::Append>(loc, vector, packed);
}
void NLJoinTranslator::scanHT(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) {
   auto scope = context.createScope();
   {
      auto forOp2 = builder.create<pgx::mlir::dsa::ForOp>(loc, mlir::TypeRange{}, vector, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType, loc);
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      auto unpacked = builder2.create<pgx::mlir::util::UnPackOp>(loc, forOp2.getInductionVar());
      orderedAttributesLeft.setValuesForColumns(context, scope, unpacked.getResults());
      Value marker = impl->markable ? unpacked.getResult(unpacked.getNumResults() - 1) : Value();
      impl->handleScanned(marker, context, builder2);
      builder2.create<pgx::mlir::dsa::YieldOp>(loc, mlir::ValueRange{});
   }
}

void NLJoinTranslator::probe(mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) {
   auto scope = context.createScope();
   impl->beforeLookup(context, builder);
   {
      auto forOp2 = builder.create<pgx::mlir::dsa::ForOp>(loc, mlir::TypeRange{}, vector, impl->getFlag(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType, loc);
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      auto unpacked = builder2.create<pgx::mlir::util::UnPackOp>(loc, forOp2.getInductionVar());
      orderedAttributesLeft.setValuesForColumns(context, scope, unpacked.getResults());
      Value markerLeft = impl->markable ? unpacked.getResult(unpacked.getNumResults() - 1) : Value();
      Value matched = evaluatePredicate(context, builder2, scope);
      impl->handleLookup(matched, markerLeft, context, builder2);
      builder2.create<pgx::mlir::dsa::YieldOp>(loc, mlir::ValueRange{});
   }
   impl->afterLookup(context, builder);
}
void NLJoinTranslator::consume(pgx::mlir::relalg::Translator* child, mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) {
   if (child == this->children[0].get()) {
      build(builder, context);
   } else if (child == this->children[1].get()) {
      probe(builder, context);
   }
}
void NLJoinTranslator::produce(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) {
   auto leftAttributes = this->requiredAttributes.intersect(children[0]->getAvailableColumns());
   orderedAttributesLeft = pgx::mlir::relalg::OrderedAttributes::fromColumns(leftAttributes);
   tupleType = orderedAttributesLeft.getTupleType(op.getContext(), impl->markable ? std::vector<Type>({mlir::IntegerType::get(op->getContext(), 64)}) : std::vector<Type>());
   vector = builder.create<pgx::mlir::dsa::CreateDS>(loc, pgx::mlir::dsa::VectorType::get(builder.getContext(), tupleType));
   children[0]->produce(context, builder);
   children[1]->produce(context, builder);
   impl->after(context, builder);
   builder.create<pgx::mlir::dsa::FreeOp>(loc, vector);
}
