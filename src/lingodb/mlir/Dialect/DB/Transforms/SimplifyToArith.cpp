#include "mlir/Dialect/Arith/IR/Arith.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <iostream>

#include "lingodb/mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <variant>
namespace {
mlir::arith::CmpIPredicateAttr convertToCmpIPred(::mlir::OpBuilder, ::mlir::db::DBCmpPredicateAttr p) {
   using namespace mlir;
   switch (p.getValue()) {
      case db::DBCmpPredicate::eq:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::eq);
      case db::DBCmpPredicate::neq:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::ne);
      case db::DBCmpPredicate::lt:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::slt);
      case db::DBCmpPredicate::gt:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::sgt);
      case db::DBCmpPredicate::lte:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::sle);
      case db::DBCmpPredicate::gte:
         return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::sge);
   }
   return mlir::arith::CmpIPredicateAttr::get(p.getContext(), arith::CmpIPredicate::sge);
}
mlir::arith::CmpFPredicateAttr convertToCmpFPred(::mlir::OpBuilder, ::mlir::db::DBCmpPredicateAttr p) {
   using namespace mlir;
   switch (p.getValue()) {
      case db::DBCmpPredicate::eq:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OEQ);
      case db::DBCmpPredicate::neq:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::ONE);
      case db::DBCmpPredicate::lt:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OLT);
      case db::DBCmpPredicate::gt:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OGT);
      case db::DBCmpPredicate::lte:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OLE);
      case db::DBCmpPredicate::gte:
         return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OGE);
   }
   return mlir::arith::CmpFPredicateAttr::get(p.getContext(), arith::CmpFPredicate::OGE);
}
mlir::Attribute convertConst(::mlir::Attribute attr, ::mlir::Value v) {
   using namespace mlir;
   std::variant<int64_t, double, std::string> parseArg;
   if (auto integerAttr = attr.dyn_cast_or_null<IntegerAttr>()) {
      if (v.getType().isIntOrIndex()) {
         return IntegerAttr::get(v.getType(), integerAttr.getInt());
      }
   } else if (auto floatAttr = attr.dyn_cast_or_null<FloatAttr>()) {
      if (v.getType().isa<::mlir::FloatType>()) {
         return FloatAttr::get(v.getType(), floatAttr.getValueAsDouble());
      }
   }
   return attr;
}
struct DBCmpToCmpI : public ::mlir::OpRewritePattern<::mlir::db::CmpOp> {
   using OpRewritePattern<::mlir::db::CmpOp>::OpRewritePattern;
   
   ::mlir::LogicalResult matchAndRewrite(::mlir::db::CmpOp op, 
                                         ::mlir::PatternRewriter &rewriter) const override {
      if (!op.getLeft().getType().isIntOrIndex() || !op.getRight().getType().isIntOrIndex()) {
         return ::mlir::failure();
      }
      
      auto pred = convertToCmpIPred(rewriter, op.getPredicateAttr());
      rewriter.replaceOpWithNewOp<::mlir::arith::CmpIOp>(op, pred, op.getLeft(), op.getRight());
      return ::mlir::success();
   }
};

struct DBCmpToCmpF : public ::mlir::OpRewritePattern<::mlir::db::CmpOp> {
   using OpRewritePattern<::mlir::db::CmpOp>::OpRewritePattern;
   
   ::mlir::LogicalResult matchAndRewrite(::mlir::db::CmpOp op, 
                                         ::mlir::PatternRewriter &rewriter) const override {
      if (!op.getLeft().getType().isa<::mlir::FloatType>() || !op.getRight().getType().isa<::mlir::FloatType>()) {
         return ::mlir::failure();
      }
      
      auto pred = convertToCmpFPred(rewriter, op.getPredicateAttr());
      rewriter.replaceOpWithNewOp<::mlir::arith::CmpFOp>(op, pred, op.getLeft(), op.getRight());
      return ::mlir::success();
   }
};

struct DBAddToAddI : public ::mlir::OpRewritePattern<::mlir::db::AddOp> {
   using OpRewritePattern<::mlir::db::AddOp>::OpRewritePattern;
   
   ::mlir::LogicalResult matchAndRewrite(::mlir::db::AddOp op, 
                                         ::mlir::PatternRewriter &rewriter) const override {
      if (!op.getType().isIntOrIndex()) {
         return ::mlir::failure();
      }
      
      rewriter.replaceOpWithNewOp<::mlir::arith::AddIOp>(op, op.getLeft(), op.getRight());
      return ::mlir::success();
   }
};

struct DBAddToAddF : public ::mlir::OpRewritePattern<::mlir::db::AddOp> {
   using OpRewritePattern<::mlir::db::AddOp>::OpRewritePattern;
   
   ::mlir::LogicalResult matchAndRewrite(::mlir::db::AddOp op, 
                                         ::mlir::PatternRewriter &rewriter) const override {
      if (!op.getType().isa<::mlir::FloatType>()) {
         return ::mlir::failure();
      }
      
      rewriter.replaceOpWithNewOp<::mlir::arith::AddFOp>(op, op.getLeft(), op.getRight());
      return ::mlir::success();
   }
};

struct DBConstToConst : public ::mlir::OpRewritePattern<::mlir::db::ConstantOp> {
   using OpRewritePattern<::mlir::db::ConstantOp>::OpRewritePattern;
   
   ::mlir::LogicalResult matchAndRewrite(::mlir::db::ConstantOp op, 
                                         ::mlir::PatternRewriter &rewriter) const override {
      auto convertedAttr = convertConst(op.getValueAttr(), op.getResult());
      if (!convertedAttr) {
         return ::mlir::failure();
      }
      
      auto typedAttr = convertedAttr.dyn_cast<::mlir::TypedAttr>();
      if (!typedAttr) {
         return ::mlir::failure();
      }
      
      rewriter.replaceOpWithNewOp<::mlir::arith::ConstantOp>(op, op.getType(), typedAttr);
      return ::mlir::success();
   }
};

class SimplifyToArith : public ::mlir::PassWrapper<SimplifyToArith, ::mlir::OperationPass<::mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "db-simplify-to-arith"; }

   public:
   void runOnOperation() override {
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<DBCmpToCmpI>(&getContext());
         patterns.insert<DBCmpToCmpF>(&getContext());
         patterns.insert<DBAddToAddI>(&getContext());
         patterns.insert<DBAddToAddF>(&getContext());
         patterns.insert<DBConstToConst>(&getContext());

         mlir::GreedyRewriteConfig config;
         config.maxIterations = 5;
         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns), config).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace mlir::db {

std::unique_ptr<Pass> createSimplifyToArithPass() { return std::make_unique<SimplifyToArith>(); }

} // end namespace mlir::db