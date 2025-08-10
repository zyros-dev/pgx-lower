#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

using namespace mlir;

namespace {

static mlir::LLVM::LLVMStructType convertTuple(TupleType tupleType, const TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      types.push_back(typeConverter.convertType(t));
   }
   return mlir::LLVM::LLVMStructType::getLiteral(tupleType.getContext(), types);
}

class PackOpLowering : public OpConversionPattern<pgx::mlir::util::PackOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::PackOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::PackOp packOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto tupleType = packOp.getTuple().getType().dyn_cast_or_null<TupleType>();
      auto structType = convertTuple(tupleType, *typeConverter);
      Value tpl = rewriter.create<LLVM::UndefOp>(packOp->getLoc(), structType);
      unsigned pos = 0;
      for (auto val : adaptor.getVals()) {
         tpl = rewriter.create<LLVM::InsertValueOp>(packOp->getLoc(), tpl, val, rewriter.getDenseI64ArrayAttr({pos++}));
      }
      rewriter.replaceOp(packOp, tpl);
      return success();
   }
};
class UndefOpLowering : public OpConversionPattern<pgx::mlir::util::UndefOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::UndefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::UndefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto ty = typeConverter->convertType(op->getResult(0).getType());
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, ty);
      return success();
   }
};
class GetTupleOpLowering : public OpConversionPattern<pgx::mlir::util::GetTupleOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::GetTupleOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::GetTupleOp getTupleOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto resType = typeConverter->convertType(getTupleOp.getVal().getType());
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(getTupleOp, resType, adaptor.getTuple(), rewriter.getDenseI64ArrayAttr({getTupleOp.getOffset()}));
      return success();
   }
};
class SizeOfOpLowering : public ConversionPattern {
   public:
   DataLayout defaultLayout;
   LLVMTypeConverter& llvmTypeConverter;
   explicit SizeOfOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, pgx::mlir::util::SizeOfOp::getOperationName(), 1, context), defaultLayout(), llvmTypeConverter(typeConverter) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      auto sizeOfOp = mlir::dyn_cast_or_null<pgx::mlir::util::SizeOfOp>(op);
      Type t = typeConverter->convertType(sizeOfOp.getType());
      const DataLayout* layout = &defaultLayout;
      if (const DataLayoutAnalysis* analysis = llvmTypeConverter.getDataLayoutAnalysis()) {
         layout = &analysis->getAbove(op);
      }
      size_t typeSize = layout->getTypeSize(t);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(op, rewriter.getI64Type(), rewriter.getI64IntegerAttr(typeSize));
      return success();
   }
};

class ToGenericMemrefOpLowering : public OpConversionPattern<pgx::mlir::util::ToGenericMemrefOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::ToGenericMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::ToGenericMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto* context = getContext();
      auto genericMemrefType = op.getRef().getType().cast<pgx::mlir::util::RefType>();
      auto i8PointerType = mlir::LLVM::LLVMPointerType::get(context);
      auto elemType = typeConverter->convertType(genericMemrefType.getElementType());
      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(elemType.getContext());
      Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), i8PointerType, adaptor.getMemref(), rewriter.getDenseI64ArrayAttr({1}));
      Value elementPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), elemPtrType, alignedPtr);
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};
class ToMemrefOpLowering : public OpConversionPattern<pgx::mlir::util::ToMemrefOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::ToMemrefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::ToMemrefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto memrefType = op.getMemref().getType().cast<MemRefType>();

      auto targetType = typeConverter->convertType(memrefType);

      auto targetPointerType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(memrefType.getElementType()).getContext());
      Value tpl = rewriter.create<LLVM::UndefOp>(op->getLoc(), targetType);

      Value elementPtr = adaptor.getRef();
      auto offset = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value deadBeefConst = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xdeadbeef));
      auto allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), targetPointerType, deadBeefConst);

      Value alignedPtr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), targetPointerType, elementPtr);
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, allocatedPtr, rewriter.getDenseI64ArrayAttr({0}));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, alignedPtr, rewriter.getDenseI64ArrayAttr({1}));
      tpl = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), targetType, tpl, offset, rewriter.getDenseI64ArrayAttr({2}));
      rewriter.replaceOp(op, tpl);
      return success();
   }
};
class IsRefValidOpLowering : public OpConversionPattern<pgx::mlir::util::IsRefValidOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::IsRefValidOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::IsRefValidOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value nullPtr = rewriter.create<::mlir::LLVM::ConstantOp>(op->getLoc(), adaptor.getRef().getType(), rewriter.getZeroAttr(adaptor.getRef().getType()));
      rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::ne, adaptor.getRef(), nullPtr);
      return success();
   }
};
class InvalidRefOpLowering : public OpConversionPattern<pgx::mlir::util::InvalidRefOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::InvalidRefOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::InvalidRefOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value nullPtr = rewriter.create<::mlir::LLVM::ConstantOp>(op->getLoc(), typeConverter->convertType(op.getType()), rewriter.getZeroAttr(typeConverter->convertType(op.getType())));
      rewriter.replaceOp(op, nullPtr);
      return success();
   }
};
class AllocaOpLowering : public OpConversionPattern<pgx::mlir::util::AllocaOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::AllocaOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::AllocaOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = allocOp->getLoc();
      auto genericMemrefType = allocOp.getRef().getType().cast<pgx::mlir::util::RefType>();
      Value entries;
      if (allocOp.getSize()) {
         entries = allocOp.getSize();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }
      auto bytesPerEntry = rewriter.create<pgx::mlir::util::SizeOfOp>(loc, rewriter.getI64Type(), genericMemrefType.getElementType());
      Value sizeInBytes = rewriter.create<mlir::arith::MulIOp>(loc, rewriter.getI64Type(), entries, bytesPerEntry);
      Value sizeInBytesI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), sizeInBytes);

      auto elemPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()).getContext());
      auto i8Type = rewriter.getI8Type();
      mlir::Value allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(loc, elemPtrType, i8Type, sizeInBytesI64, 0);
      rewriter.replaceOp(allocOp, allocatedElementPtr);

      return success();
   }
};
class AllocOpLowering : public OpConversionPattern<pgx::mlir::util::AllocOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::AllocOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::AllocOp allocOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = allocOp->getLoc();

      auto genericMemrefType = allocOp.getRef().getType().cast<pgx::mlir::util::RefType>();
      Value entries;
      if (allocOp.getSize()) {
         entries = allocOp.getSize();
      } else {
         int64_t staticSize = 1;
         entries = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(staticSize));
      }

      auto bytesPerEntry = rewriter.create<pgx::mlir::util::SizeOfOp>(loc, rewriter.getI64Type(), genericMemrefType.getElementType());
      Value sizeInBytes = rewriter.create<mlir::arith::MulIOp>(loc, rewriter.getI64Type(), entries, bytesPerEntry);
      Value sizeInBytesI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), sizeInBytes);

      auto mallocFunc = LLVM::lookupOrCreateMallocFn(allocOp->getParentOfType<ModuleOp>(), rewriter.getI64Type());
      auto results = rewriter.create<LLVM::CallOp>(loc, LLVM::LLVMPointerType::get(rewriter.getContext()), SymbolRefAttr::get(*mallocFunc), sizeInBytesI64).getResults();
      mlir::Value castedPointer = rewriter.create<LLVM::BitcastOp>(loc, LLVM::LLVMPointerType::get(typeConverter->convertType(genericMemrefType.getElementType()).getContext()), results[0]);
      rewriter.replaceOp(allocOp, castedPointer);

      return success();
   }
};
class DeAllocOpLowering : public OpConversionPattern<pgx::mlir::util::DeAllocOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::DeAllocOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::DeAllocOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());
      Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()), adaptor.getRef());
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange(), SymbolRefAttr::get(*freeFunc), casted);
      return success();
   }
};

class StoreOpLowering : public OpConversionPattern<pgx::mlir::util::StoreOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::StoreOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.getRef();
      if (adaptor.getIdx()) {
         auto i8Type = rewriter.getI8Type();
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elementPtr.getType(), i8Type, elementPtr, adaptor.getIdx());
      }
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getVal(), elementPtr);
      return success();
   }
};
class LoadOpLowering : public OpConversionPattern<pgx::mlir::util::LoadOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::LoadOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value elementPtr = adaptor.getRef();
      if (adaptor.getIdx()) {
         auto i8Type = rewriter.getI8Type();
         elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), elementPtr.getType(), i8Type, elementPtr, adaptor.getIdx());
      }
      auto resultType = typeConverter->convertType(op->getResult(0).getType());
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType, elementPtr);
      return success();
   }
};
class CastOpLowering : public OpConversionPattern<pgx::mlir::util::GenericMemrefCastOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::GenericMemrefCastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::GenericMemrefCastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetRefType = op.getRes().getType().cast<pgx::mlir::util::RefType>();
      auto targetElemType = typeConverter->convertType(targetRefType.getElementType());
      Value casted = rewriter.create<LLVM::BitcastOp>(op->getLoc(), LLVM::LLVMPointerType::get(targetElemType.getContext()), adaptor.getVal());
      rewriter.replaceOp(op, casted);
      return success();
   }
};
class TupleElementPtrOpLowering : public OpConversionPattern<pgx::mlir::util::TupleElementPtrOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::TupleElementPtrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::TupleElementPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetMemrefType = op.getType().cast<pgx::mlir::util::RefType>();
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(targetMemrefType.getElementType()).getContext());
      Value zero = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      Value structIdx = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(op.getIdx()));
      auto structElemType = typeConverter->convertType(targetMemrefType.getElementType());
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, structElemType, adaptor.getRef(), ValueRange({zero, structIdx}));
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};
class ArrayElementPtrOpLowering : public OpConversionPattern<pgx::mlir::util::ArrayElementPtrOp> {
   public:
   using OpConversionPattern<pgx::mlir::util::ArrayElementPtrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::ArrayElementPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto targetMemrefType = op.getType().cast<pgx::mlir::util::RefType>();
      auto targetPtrType = mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(targetMemrefType.getElementType()).getContext());
      auto arrayElemType = typeConverter->convertType(targetMemrefType.getElementType());
      Value elementPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), targetPtrType, arrayElemType, adaptor.getRef(), adaptor.getIdx());
      rewriter.replaceOp(op, elementPtr);
      return success();
   }
};

class CreateVarLenLowering : public OpConversionPattern<pgx::mlir::util::CreateVarLen> {
   public:
   using OpConversionPattern<pgx::mlir::util::CreateVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::CreateVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Type i128Ty = rewriter.getIntegerType(128);
      Value lazymask = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0x80000000));
      Value lazylen = rewriter.create<mlir::LLVM::OrOp>(op->getLoc(), lazymask, adaptor.getLen());
      Value asI128 = rewriter.create<mlir::LLVM::ZExtOp>(op->getLoc(), i128Ty, lazylen);
      Value rawPtr = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), i128Ty, adaptor.getRef());
      auto const64 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 64));
      auto shlPtr = rewriter.create<mlir::LLVM::ShlOp>(op->getLoc(), rawPtr, const64);
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, asI128, shlPtr);
      return success();
   }
};
class CreateConstVarLenLowering : public OpConversionPattern<pgx::mlir::util::CreateConstVarLen> {
   public:
   using OpConversionPattern<pgx::mlir::util::CreateConstVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::CreateConstVarLen op, OpAdaptor adaptor,
                                 ConversionPatternRewriter& rewriter) const override {
      size_t len = op.getStr().size();

      mlir::Type i128Ty = rewriter.getIntegerType(128);
      mlir::Value p1, p2;

      uint64_t first4 = 0;
      memcpy(&first4, op.getStr().data(), std::min(4ul, len));
      size_t c1 = (first4 << 32) | len;
      p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, c1));
      if (len <= 12) {
         uint64_t last8 = 0;
         if (len > 4) {
            memcpy(&last8, op.getStr().data() + 4, std::min(8ul, len - 4));
         }
         p2 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, last8));
      } else {
         static size_t globalStrConstId = 0;
         mlir::LLVM::GlobalOp globalOp;
         {
            std::string name = "global_str_const_" + std::to_string(globalStrConstId++);
            auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            globalOp = rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(), len), true, mlir::LLVM::Linkage::Private, name, op.getStrAttr());
         }
         auto ptr = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), globalOp);
         p2 = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), i128Ty, ptr);
      }
      auto const64 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), i128Ty, rewriter.getIntegerAttr(i128Ty, 64));
      auto shlp2 = rewriter.create<mlir::LLVM::ShlOp>(op->getLoc(), p2, const64);
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, p1, shlp2);
      return success();
   }
};

class VarLenGetLenLowering : public OpConversionPattern<pgx::mlir::util::VarLenGetLen> {
   public:
   using OpConversionPattern<pgx::mlir::util::VarLenGetLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::VarLenGetLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value len = rewriter.create<LLVM::TruncOp>(op->getLoc(), rewriter.getI64Type(), adaptor.getVarlen());
      Value mask = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x7FFFFFFF));
      Value castedLen = rewriter.create<LLVM::AndOp>(op->getLoc(), len, mask);

      rewriter.replaceOp(op, castedLen);
      return success();
   }
};
class Hash64Lowering : public OpConversionPattern<pgx::mlir::util::Hash64> {
   public:
   using OpConversionPattern<pgx::mlir::util::Hash64>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::Hash64 op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value p1 = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(11400714819323198549ull));
      Value m1 = rewriter.create<LLVM::MulOp>(op->getLoc(), p1, adaptor.getVal());
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), m1);
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), m1, reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashCombineLowering: public OpConversionPattern<pgx::mlir::util::HashCombine> {
   public:
   using OpConversionPattern<pgx::mlir::util::HashCombine>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::HashCombine op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value reversed = rewriter.create<mlir::LLVM::ByteSwapOp>(op->getLoc(), adaptor.getH1());
      Value result = rewriter.create<LLVM::XOrOp>(op->getLoc(), adaptor.getH2(), reversed);
      rewriter.replaceOp(op, result);
      return success();
   }
};
class HashVarLenLowering : public OpConversionPattern<pgx::mlir::util::HashVarLen> {
   public:
   using OpConversionPattern<pgx::mlir::util::HashVarLen>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::HashVarLen op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto fn = LLVM::lookupOrCreateFn(op->getParentOfType<ModuleOp>(), "hashVarLenData", {rewriter.getIntegerType(128)}, rewriter.getI64Type());
      auto result = rewriter.create<LLVM::CallOp>(op->getLoc(), rewriter.getI64Type(), SymbolRefAttr::get(*fn), adaptor.getVal()).getResult();
      rewriter.replaceOp(op, result);
      return success();
   }
};

class FilterTaggedPtrLowering : public OpConversionPattern<pgx::mlir::util::FilterTaggedPtr> {
   public:
   using OpConversionPattern<pgx::mlir::util::FilterTaggedPtr>::OpConversionPattern;
   LogicalResult matchAndRewrite(pgx::mlir::util::FilterTaggedPtr op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto tagMask = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0xffff000000000000ull));
      auto ptrMask = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0x0000ffffffffffffull));
      Value maskedHash = rewriter.create<LLVM::AndOp>(loc, adaptor.getHash(), tagMask);
      Value ptrAsInt = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), adaptor.getRef());
      Value maskedPtr = rewriter.create<LLVM::AndOp>(loc, ptrAsInt, ptrMask);
      maskedPtr = rewriter.create<LLVM::IntToPtrOp>(loc, adaptor.getRef().getType(), maskedPtr);
      Value ored = rewriter.create<LLVM::OrOp>(loc, ptrAsInt, maskedHash);
      Value contained = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ored, ptrAsInt);
      Value nullPtr = rewriter.create<::mlir::LLVM::ConstantOp>(loc, adaptor.getRef().getType(), rewriter.getZeroAttr(adaptor.getRef().getType()));

      Value filtered = rewriter.create<LLVM::SelectOp>(loc, contained, maskedPtr, nullPtr);
      rewriter.replaceOp(op, filtered);
      return success();
   }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct UtilToLLVMLoweringPass
   : public PassWrapper<UtilToLLVMLoweringPass, OperationPass<ModuleOp>> {
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
   }
   void runOnOperation() final;
};

void UtilToLLVMLoweringPass::runOnOperation() {
   LLVMConversionTarget target(getContext());
   RewritePatternSet patterns(&getContext());

   LLVMTypeConverter typeConverter(&getContext());
   pgx::mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);

   if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
   }
}
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> pgx::mlir::util::createUtilToLLVMPass() {
    return std::make_unique<UtilToLLVMLoweringPass>();
}

void pgx::mlir::util::populateUtilToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
   typeConverter.addConversion([&](::mlir::TupleType tupleType) -> ::mlir::LLVM::LLVMStructType {
      return convertTuple(tupleType, typeConverter);
   });
   typeConverter.addConversion([&](pgx::mlir::util::RefType genericMemrefType) -> ::mlir::Type {
      return ::mlir::LLVM::LLVMPointerType::get(typeConverter.convertType(genericMemrefType.getElementType()).getContext());
   });
   typeConverter.addConversion([&](pgx::mlir::util::VarLen32Type varLen32Type) -> ::mlir::IntegerType {
      ::mlir::MLIRContext* context = &typeConverter.getContext();
      return ::mlir::IntegerType::get(context, 128);
   });
   patterns.add<CastOpLowering>(typeConverter, patterns.getContext());
   patterns.add<SizeOfOpLowering>(typeConverter, patterns.getContext());
   patterns.add<GetTupleOpLowering>(typeConverter, patterns.getContext());
   patterns.add<UndefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<PackOpLowering>(typeConverter, patterns.getContext());
   patterns.add<AllocOpLowering>(typeConverter, patterns.getContext());
   patterns.add<AllocaOpLowering>(typeConverter, patterns.getContext());
   patterns.add<DeAllocOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ArrayElementPtrOpLowering>(typeConverter, patterns.getContext());
   patterns.add<TupleElementPtrOpLowering>(typeConverter, patterns.getContext());

   patterns.add<ToGenericMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<ToMemrefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<IsRefValidOpLowering>(typeConverter, patterns.getContext());
   patterns.add<InvalidRefOpLowering>(typeConverter, patterns.getContext());
   patterns.add<StoreOpLowering>(typeConverter, patterns.getContext());
   patterns.add<LoadOpLowering>(typeConverter, patterns.getContext());
   patterns.add<CreateVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<CreateConstVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<VarLenGetLenLowering>(typeConverter, patterns.getContext());
   patterns.add<HashCombineLowering>(typeConverter, patterns.getContext());
   patterns.add<Hash64Lowering>(typeConverter, patterns.getContext());
   patterns.add<HashVarLenLowering>(typeConverter, patterns.getContext());
   patterns.add<FilterTaggedPtrLowering>(typeConverter, patterns.getContext());
}
