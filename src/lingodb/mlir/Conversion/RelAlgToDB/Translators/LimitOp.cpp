#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"

class LimitTranslator : public mlir::relalg::Translator {
   mlir::relalg::LimitOp limitOp;
   ::mlir::Value counter;

   public:
   LimitTranslator(mlir::relalg::LimitOp limitOp) : mlir::relalg::Translator(limitOp), limitOp(limitOp) {}

   virtual void consume(mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto one = builder.create<mlir::arith::ConstantIntOp>(limitOp.getLoc(), 1, 64);
      ::mlir::Value loadedCounter = builder.create<mlir::util::LoadOp>(limitOp.getLoc(), builder.getI64Type(), counter, ::mlir::Value());
      ::mlir::Value addedCounter = builder.create<mlir::arith::AddIOp>(limitOp.getLoc(), loadedCounter, one);
      ::mlir::Value upper = builder.create<mlir::arith::ConstantIntOp>(limitOp.getLoc(), limitOp.getRows(), 64);
      ::mlir::Value considerTuple = builder.create<mlir::arith::CmpIOp>(limitOp.getLoc(), mlir::arith::CmpIPredicate::ule, addedCounter, upper);
      auto ifOp = builder.create<mlir::scf::IfOp>(
         limitOp->getLoc(), considerTuple,
         [&](mlir::OpBuilder& b, mlir::Location loc) {
            // Then branch
            consumer->consume(this, b, context);
            b.create<mlir::scf::YieldOp>(loc);
         }
      );
      builder.create<mlir::util::StoreOp>(limitOp.getLoc(), addedCounter, counter, ::mlir::Value());
   }

   virtual void produce(mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      counter = builder.create<mlir::util::AllocaOp>(limitOp.getLoc(), mlir::util::RefType::get(builder.getContext(), builder.getI64Type()), ::mlir::Value());
      ::mlir::Value zero = builder.create<mlir::db::ConstantOp>(limitOp.getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(0));
      builder.create<mlir::util::StoreOp>(limitOp.getLoc(), zero, counter, ::mlir::Value());
      children[0]->produce(context, builder);
   }

   virtual ~LimitTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createLimitTranslator(mlir::relalg::LimitOp limitOp) {
   return std::make_unique<LimitTranslator>(limitOp);
}