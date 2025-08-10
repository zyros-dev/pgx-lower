#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "execution/logging.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace pgx::mlir::util;

::mlir::LogicalResult pgx::mlir::util::UnPackOp::verify() {
   pgx::mlir::util::UnPackOp& unPackOp = *this;
   if (auto tupleType = unPackOp.getTuple().getType().dyn_cast_or_null<::mlir::TupleType>()) {
      if (tupleType.getTypes().size() != unPackOp.getVals().size()) {
         unPackOp.emitOpError("must unpack exactly as much as entries in tuple");
         unPackOp.dump();
         return failure();
      }
      for (size_t i = 0; i < tupleType.getTypes().size(); i++) {
         if (tupleType.getTypes()[i] != unPackOp.getVals()[i].getType()) {
            unPackOp.emitOpError("types must match during unpacking");
            unPackOp.dump();
            return failure();
         }
      }
   } else {
      unPackOp.emitOpError("must be tupletype");
      return failure();
   }
   return success();
}
::mlir::LogicalResult pgx::mlir::util::PackOp::verify() {
   pgx::mlir::util::PackOp& packOp = *this;
   if (auto tupleType = packOp.getTuple().getType().dyn_cast_or_null<::mlir::TupleType>()) {
      if (tupleType.getTypes().size() != packOp.getVals().size()) {
         packOp.emitOpError("must pack exactly as much as entries in tuple");
         packOp.dump();
         return failure();
      }
      for (size_t i = 0; i < tupleType.getTypes().size(); i++) {
         if (tupleType.getTypes()[i] != packOp.getVals()[i].getType()) {
            packOp.emitOpError("types must match during unpacking");
            packOp.dump();
            return failure();
         }
      }
   } else {
      packOp.emitOpError("must be tupletype");
      return failure();
   }
   return success();
}

LogicalResult pgx::mlir::util::UnPackOp::canonicalize(pgx::mlir::util::UnPackOp unPackOp, mlir::PatternRewriter& rewriter) {
   auto tuple = unPackOp.getTuple();
   if (auto* tupleCreationOp = tuple.getDefiningOp()) {
      if (auto packOp = dyn_cast_or_null<PackOp>(tupleCreationOp)) {
         rewriter.replaceOp(unPackOp.getOperation(), packOp.getVals());
         return success();
      }
   }
   std::vector<Value> vals;
   vals.reserve(unPackOp.getNumResults());
   for (unsigned i = 0; i < unPackOp.getNumResults(); i++) {
      auto ty = unPackOp.getResultTypes()[i];
      vals.push_back(rewriter.create<GetTupleOp>(unPackOp.getLoc(), ty, tuple, i));
   }
   rewriter.replaceOp(unPackOp.getOperation(), vals);
   return success();
}

LogicalResult pgx::mlir::util::GetTupleOp::canonicalize(pgx::mlir::util::GetTupleOp op, mlir::PatternRewriter& rewriter) {
   if (auto* tupleCreationOp = op.getTuple().getDefiningOp()) {
      if (auto packOp = dyn_cast_or_null<PackOp>(tupleCreationOp)) {
         rewriter.replaceOp(op.getOperation(), packOp.getOperand(op.getOffset()));
         return success();
      }
      if (auto selOp = dyn_cast_or_null<arith::SelectOp>(tupleCreationOp)) {
         auto sel1 = rewriter.create<GetTupleOp>(op.getLoc(), op.getVal().getType(), selOp.getTrueValue(), op.getOffset());
         auto sel2 = rewriter.create<GetTupleOp>(op.getLoc(), op.getVal().getType(), selOp.getFalseValue(), op.getOffset());
         rewriter.replaceOpWithNewOp<arith::SelectOp>(op, selOp.getCondition(), sel1, sel2);
         return success();
      }
      if (auto loadOp = dyn_cast_or_null<LoadOp>(tupleCreationOp)) {
         OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(loadOp);
         auto base = loadOp.getRef();
         if (auto idx = loadOp.getIdx()) {
            base = rewriter.create<ArrayElementPtrOp>(loadOp.getLoc(), base.getType(), base, idx);
         }

         auto elemTy = op.getResult().getType();
         auto elemRefTy = RefType::get(elemTy);
         auto tep = rewriter.create<TupleElementPtrOp>(loadOp.getLoc(), elemRefTy, base, op.getOffset());
         auto newLoad = rewriter.create<LoadOp>(loadOp.getLoc(), tep);
         rewriter.replaceOp(op.getOperation(), newLoad.getResult());
         return success();
      }
   }
   return failure();
}

LogicalResult pgx::mlir::util::StoreOp::canonicalize(pgx::mlir::util::StoreOp op, mlir::PatternRewriter& rewriter) {
   if (auto ty = op.getVal().getType().dyn_cast_or_null<::mlir::TupleType>()) {
      auto base = op.getRef();
      if (auto idx = op.getIdx()) {
         base = rewriter.create<ArrayElementPtrOp>(op.getLoc(), base.getType(), base, idx);
      }
      for (size_t i = 0; i < ty.size(); i++) {
         auto elemRefTy = RefType::get(ty.getType(i));
         auto gt = rewriter.create<GetTupleOp>(op.getLoc(), ty.getType(i), op.getVal(), i);
         auto tep = rewriter.create<TupleElementPtrOp>(op.getLoc(), elemRefTy, base, i);
         rewriter.create<StoreOp>(op.getLoc(), gt, tep, Value());
      }
      rewriter.eraseOp(op);
      return success();
   }
   return failure();
}

void pgx::mlir::util::LoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects) {
   if (!getOperation()->hasAttr("nosideffect")) {
      effects.emplace_back(MemoryEffects::Read::get());
   }
}
::mlir::LogicalResult pgx::mlir::util::TupleElementPtrOp::verify() {
   pgx::mlir::util::TupleElementPtrOp& op = *this;
   auto resElementType = op.getType().getElementType();
   auto ptrTupleType = op.getRef().getType().cast<RefType>().getElementType().cast<::mlir::TupleType>();
   auto ptrElementType = ptrTupleType.getTypes()[op.getIdx()];
   if (resElementType != ptrElementType) {
      op.emitOpError("Element types do not match");
      OpPrintingFlags flags;
      flags.assumeVerified();
      op->print(llvm::outs(), flags);
      return failure();
   }
   return success();
}
#define GET_OP_CLASSES
#include "mlir/Dialect/Util/IR/UtilOps.cpp.inc"
