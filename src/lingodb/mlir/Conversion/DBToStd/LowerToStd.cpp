#include "lingodb/mlir-support/parsing.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "lingodb/mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "lingodb/mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "lingodb/mlir/Dialect/DB/Passes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "pgx-lower/utility/logging.h"
#include "runtime-defs/NumericRuntime.h"
#include "runtime-defs/StringRuntime.h"

#include <catalog/pg_type_d.h>
#include <utils/fmgroids.h>
#include <lingodb/mlir/Dialect/util/FunctionHelper.h>
#include <lingodb/utility/mlir_to_postgres.h>
#include <string>
#include <type_traits>

using namespace mlir;

namespace {
struct DBToStdLoweringPass
   : public PassWrapper<DBToStdLoweringPass, OperationPass<ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "to-arrow-std"; }

   DBToStdLoweringPass() {}
   void getDependentDialects(DialectRegistry& registry) const override {
      registry.insert<LLVM::LLVMDialect, mlir::db::DBDialect, scf::SCFDialect, mlir::cf::ControlFlowDialect, util::UtilDialect, memref::MemRefDialect, arith::ArithDialect>();
   }
   void runOnOperation() final;
};
static TupleType convertTuple(TupleType tupleType, TypeConverter& typeConverter) {
   std::vector<Type> types;
   for (auto t : tupleType.getTypes()) {
      Type converted = typeConverter.convertType(t);
      converted = converted ? converted : t;
      types.push_back(converted);
   }
   return TupleType::get(tupleType.getContext(), TypeRange(types));
}
} // end anonymous namespace
static bool hasDBType(TypeConverter& converter, TypeRange types) {
   return llvm::any_of(types, [&converter](mlir::Type t) { auto converted = converter.convertType(t);return converted&&converted!=t; });
}

static bool isNumericCarrierType(mlir::Type type);
static mlir::Value
combineNullFlags(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value leftNull, mlir::Value rightNull);

static bool isNullableDbValueType(mlir::Type type) {
    return type.isa<mlir::db::NullableType>()
           || (mlir::db::isPgValueType(type) && mlir::db::getPgNullability(type) == mlir::db::PgNullability::Maybe);
}

static bool isPgIntegerValueType(mlir::Type type) {
    type = getBaseType(type);
    return mlir::isa<mlir::db::PgInt2Type, mlir::db::PgInt4Type, mlir::db::PgInt8Type>(type);
}

static bool isPgFloatValueType(mlir::Type type) {
    type = getBaseType(type);
    return mlir::isa<mlir::db::PgFloat4Type, mlir::db::PgFloat8Type>(type);
}

static bool isPgStringValueType(mlir::Type type) {
    type = getBaseType(type);
    return mlir::isa<mlir::db::PgTextType, mlir::db::PgVarcharType, mlir::db::PgBpcharType>(type);
}

static mlir::Type getNonNullablePgBaseType(mlir::Type type) {
    type = getBaseType(type);
    if (mlir::db::isPgValueType(type)) {
        return mlir::db::withPgNullability(type, mlir::db::PgNullability::Never);
    }
    return type;
}

static bool isPgDateValueType(mlir::Type type) {
    return mlir::isa<mlir::db::PgDateType>(getNonNullablePgBaseType(type));
}

static bool isPgInt4ValueType(mlir::Type type) {
    return mlir::isa<mlir::db::PgInt4Type>(getNonNullablePgBaseType(type));
}

static bool isPgTimestampValueType(mlir::Type type) {
    return mlir::isa<mlir::db::PgTimestampType>(getNonNullablePgBaseType(type));
}

static constexpr int64_t kPgMicrosecondsPerDay = 86400000000LL;

static mlir::Value pgDateDaysToTimestampMicros(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value days) {
    if (days.getType().isInteger(32)) {
        days = builder.create<mlir::arith::ExtSIOp>(loc, builder.getI64Type(), days);
    }
    mlir::Value multiplier = builder.create<mlir::arith::ConstantIntOp>(loc, kPgMicrosecondsPerDay, 64);
    return builder.create<mlir::arith::MulIOp>(loc, days, multiplier);
}

static int getSignedIntegerCarrierWidth(mlir::Type type) {
    if (auto width = getIntegerWidth(type, false)) {
        return width;
    }
    if (!isPgIntegerValueType(type)) {
        return 0;
    }
    return getIntegerWidth(mlir::db::getPgPhysicalCarrierType(type), false);
}

static mlir::FloatType getFloatCarrierType(mlir::Type type) {
    if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
        return floatType;
    }
    if (!mlir::db::isPgValueType(type)) {
        return {};
    }
    return mlir::dyn_cast_or_null<mlir::FloatType>(mlir::db::getPgPhysicalCarrierType(type));
}

template<class OperandType>
static bool supportsPhysicalBinOp(mlir::Type type) {
    type = getBaseType(type);
    if (type.isa<OperandType>()) {
        return true;
    }
    if constexpr (std::is_same_v<OperandType, mlir::IntegerType>) {
        return isPgIntegerValueType(type);
    }
    if constexpr (std::is_same_v<OperandType, mlir::FloatType>) {
        return isPgFloatValueType(type);
    }
    return false;
}

static mlir::Value scalarConstant(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type, int64_t value) {
    if (type.isIndex()) {
        return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
    }
    if (auto integerType = type.dyn_cast_or_null<mlir::IntegerType>()) {
        return builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(integerType, value));
    }
    if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
        return builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(floatType, static_cast<double>(value)));
    }
    return {};
}

static mlir::Value
safePayloadOr(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value payload, mlir::Value isNull, int64_t fallback) {
    if (!isNull) {
        return payload;
    }
    mlir::Value fallbackValue = scalarConstant(builder, loc, payload.getType(), fallback);
    if (!fallbackValue) {
        return payload;
    }
    return builder.create<mlir::arith::SelectOp>(loc, isNull, fallbackValue, payload);
}

static mlir::Value
castPhysicalScalar(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::Type targetType) {
    if (value.getType() == targetType) {
        return value;
    }
    auto sourceInt = value.getType().dyn_cast_or_null<mlir::IntegerType>();
    auto targetInt = targetType.dyn_cast_or_null<mlir::IntegerType>();
    if (sourceInt && targetInt) {
        if (sourceInt.getWidth() < targetInt.getWidth()) {
            return builder.create<mlir::arith::ExtSIOp>(loc, targetType, value);
        }
        return builder.create<mlir::arith::TruncIOp>(loc, targetType, value);
    }
    auto sourceFloat = value.getType().dyn_cast_or_null<mlir::FloatType>();
    auto targetFloat = targetType.dyn_cast_or_null<mlir::FloatType>();
    if (sourceFloat && targetFloat) {
        if (sourceFloat.getWidth() < targetFloat.getWidth()) {
            return builder.create<mlir::arith::ExtFOp>(loc, targetType, value);
        }
        return builder.create<mlir::arith::TruncFOp>(loc, targetType, value);
    }
    return value;
}

struct NullableOperand {
    mlir::Value payload;
    mlir::Value isNull;
};

static NullableOperand unwrapNullableOperand(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value) {
    if (auto tupleType = mlir::dyn_cast_or_null<mlir::TupleType>(value.getType());
        tupleType && tupleType.size() == 2 && tupleType.getType(0).isInteger(1))
    {
        auto unpacked = builder.create<mlir::util::UnPackOp>(loc, value);
        return {unpacked.getVals()[1], unpacked.getVals()[0]};
    }
    return {value, mlir::Value()};
}

struct PgTypeSnapshot {
    uint32_t oid;
    int32_t typmod;
    uint32_t collation;
};

static mlir::Type getOriginalPgBaseType(mlir::Type type) {
    if (auto nullableType = mlir::dyn_cast<mlir::db::NullableType>(type)) {
        return nullableType.getType();
    }
    if (auto tupleType = mlir::dyn_cast<mlir::TupleType>(type)) {
        if (tupleType.getTypes().size() == 2 && tupleType.getTypes()[0].isInteger(1)) {
            return tupleType.getTypes()[1];
        }
    }
    return type;
}

static PgTypeSnapshot getOriginalPgTypeSnapshot(mlir::Type type) {
    mlir::Type baseType = getOriginalPgBaseType(type);
    if (mlir::db::isPgValueType(baseType)) {
        return {mlir::db::getPgTypeOid(baseType), mlir::db::getPgTypmod(baseType), mlir::db::getPgCollation(baseType)};
    }
    return {InvalidOid, -1, InvalidOid};
}

struct PgTypeSnapshotAttrs {
    mlir::ArrayAttr oids;
    mlir::ArrayAttr typmods;
    mlir::ArrayAttr collations;
};

static PgTypeSnapshotAttrs
getOriginalPgTypeSnapshotAttrs(mlir::TupleType tupleType, mlir::ConversionPatternRewriter& rewriter) {
    llvm::SmallVector<mlir::Attribute> oidAttrs;
    llvm::SmallVector<mlir::Attribute> typmodAttrs;
    llvm::SmallVector<mlir::Attribute> collationAttrs;
    oidAttrs.reserve(tupleType.size());
    typmodAttrs.reserve(tupleType.size());
    collationAttrs.reserve(tupleType.size());
    for (mlir::Type fieldType : tupleType.getTypes()) {
        PgTypeSnapshot snapshot = getOriginalPgTypeSnapshot(fieldType);
        oidAttrs.push_back(rewriter.getI32IntegerAttr(snapshot.oid));
        typmodAttrs.push_back(rewriter.getI32IntegerAttr(snapshot.typmod));
        collationAttrs.push_back(rewriter.getI32IntegerAttr(snapshot.collation));
    }
    return {rewriter.getArrayAttr(oidAttrs), rewriter.getArrayAttr(typmodAttrs), rewriter.getArrayAttr(collationAttrs)};
}

static void setOriginalPgTypeAttrs(mlir::Operation* op, const char* prefix, mlir::TupleType tupleType,
                                   mlir::ConversionPatternRewriter& rewriter) {
    PgTypeSnapshotAttrs attrs = getOriginalPgTypeSnapshotAttrs(tupleType, rewriter);
    op->setAttr(std::string(prefix) + "_oids", attrs.oids);
    op->setAttr(std::string(prefix) + "_typmods", attrs.typmods);
    op->setAttr(std::string(prefix) + "_collations", attrs.collations);
}

static uint32_t pgStringCompareFunctionOid(mlir::Type type, mlir::db::DBCmpPredicate predicate) {
    type = getBaseType(type);
    if (mlir::isa<mlir::db::PgBpcharType>(type)) {
        switch (predicate) {
        case mlir::db::DBCmpPredicate::eq: return F_BPCHAREQ;
        case mlir::db::DBCmpPredicate::neq: return F_BPCHARNE;
        case mlir::db::DBCmpPredicate::lt: return F_BPCHARLT;
        case mlir::db::DBCmpPredicate::gt: return F_BPCHARGT;
        case mlir::db::DBCmpPredicate::lte: return F_BPCHARLE;
        case mlir::db::DBCmpPredicate::gte: return F_BPCHARGE;
        }
    }
    return InvalidOid;
}

static uint32_t pgStringHashFunctionOid(mlir::Type type) {
    type = getBaseType(type);
    if (mlir::isa<mlir::db::PgBpcharType>(type)) {
        return F_HASHBPCHAR;
    }
    if (mlir::isa<mlir::db::PgTextType, mlir::db::PgVarcharType>(type)) {
        return F_HASHTEXT;
    }
    return InvalidOid;
}

template <class Op>
class SimpleTypeConversionPattern : public ConversionPattern {
   mlir::LogicalResult safelyMoveRegion(ConversionPatternRewriter& rewriter, mlir::TypeConverter& typeConverter, mlir::Region& source, mlir::Region& target) const {
      rewriter.inlineRegionBefore(source, target, target.end());
      {
         if (!target.empty()) {
            source.push_back(new Block);
            std::vector<mlir::Location> locs;
            for (size_t i = 0; i < target.front().getArgumentTypes().size(); i++) {
               locs.push_back(target.front().getArgument(i).getLoc());
            }
            source.front().addArguments(target.front().getArgumentTypes(), locs);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&source.front());
            rewriter.create<mlir::dsa::YieldOp>(rewriter.getUnknownLoc());
         }
      }
      if (failed(rewriter.convertRegionTypes(&target, typeConverter))) {
         return rewriter.notifyMatchFailure(source.getParentOp(), "could not convert body types");
      }
      return success();
   }

   public:
   explicit SimpleTypeConversionPattern(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, Op::getOperationName(), 1, context) {}

   LogicalResult
   matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                   ConversionPatternRewriter& rewriter) const override {
      llvm::SmallVector<mlir::Type> convertedTypes;
      assert(typeConverter->convertTypes(op->getResultTypes(), convertedTypes).succeeded());
      auto newOp = rewriter.create<Op>(op->getLoc(), convertedTypes, ValueRange(operands), op->getAttrs());
      if constexpr (std::is_same_v<Op, mlir::dsa::CreateDS>) {
          auto createOp = mlir::cast<mlir::dsa::CreateDS>(op);
          if (auto genericType = mlir::dyn_cast<mlir::dsa::GenericIterableType>(createOp.getDs().getType())) {
              if (genericType.getIteratorName() == "pgsort_iterator") {
                  if (auto tupleType = mlir::dyn_cast<mlir::TupleType>(genericType.getElementType())) {
                      setOriginalPgTypeAttrs(newOp, "pgx_original_type", tupleType, rewriter);
                  }
              }
          } else if (auto joinHtType = mlir::dyn_cast<mlir::dsa::JoinHashtableType>(createOp.getDs().getType())) {
              setOriginalPgTypeAttrs(newOp, "pgx_original_key_type", joinHtType.getKeyType(), rewriter);
              setOriginalPgTypeAttrs(newOp, "pgx_original_val_type", joinHtType.getValType(), rewriter);
          } else if (auto aggrHtType = mlir::dyn_cast<mlir::dsa::AggregationHashtableType>(createOp.getDs().getType()))
          {
              setOriginalPgTypeAttrs(newOp, "pgx_original_key_type", aggrHtType.getKeyType(), rewriter);
              setOriginalPgTypeAttrs(newOp, "pgx_original_val_type", aggrHtType.getValType(), rewriter);
          }
      }
      for (size_t i = 0; i < op->getNumRegions(); i++) {
         if (safelyMoveRegion(rewriter, const_cast<TypeConverter&>(*typeConverter), op->getRegion(i), newOp->getRegion(i)).failed()) {
            return failure();
         }
      }
      rewriter.replaceOp(op, newOp->getResults());
      return success();
   }
};
class AtLowering : public OpConversionPattern<mlir::dsa::At> {
   public:
   using OpConversionPattern<mlir::dsa::At>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::dsa::At atOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = atOp->getLoc();
      auto t = atOp.getType(0);
      if (typeConverter->isLegal(t)) {
         rewriter.modifyOpInPlace(atOp, [&]() {
            atOp->setOperands(adaptor.getOperands());
         });
         return mlir::success();
      }
      auto* context = getContext();
      mlir::Type arrowPhysicalType = typeConverter->convertType(t);
      if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
          arrowPhysicalType = dateType.getUnit() == mlir::db::DateUnitAttr::day ? mlir::IntegerType::get(context, 32)
                                                                                : mlir::IntegerType::get(context, 64);
      }
      llvm::SmallVector<mlir::Type> types;
      types.push_back(arrowPhysicalType);
      if (atOp.getValid()) {
         types.push_back(rewriter.getI1Type());
      }
      std::vector<mlir::Value> values;
      auto newAtOp = rewriter.create<mlir::dsa::At>(loc, types, adaptor.getCollection(), atOp.getPos());
      values.push_back(newAtOp.getVal());
      if (atOp.getValid()) {
         values.push_back(newAtOp.getValid());
      }
      if (t.isa<mlir::db::DateType, mlir::db::TimestampType>()) {
         if (values[0].getType() != rewriter.getI64Type()) {
            values[0] = rewriter.create<mlir::arith::ExtSIOp>(loc, rewriter.getI64Type(), values[0]);
         }
         size_t multiplier = 1;
         if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
            multiplier = dateType.getUnit() == mlir::db::DateUnitAttr::day ? 86400000000000 : 1000000;
         } else if (auto timeStampType = t.dyn_cast_or_null<mlir::db::TimestampType>()) {
            multiplier = 1000;  // microseconds to nanoseconds
         }
         if (multiplier != 1) {
            mlir::Value multiplierConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, multiplier, 64);
            values[0] = rewriter.create<mlir::arith::MulIOp>(loc, values[0], multiplierConst);
         }
      } else if (t.isa<mlir::db::IntervalType>()) {
         if (values[0].getType() != rewriter.getI64Type()) {
            values[0] = rewriter.create<mlir::arith::ExtSIOp>(loc, rewriter.getI64Type(), values[0]);
         }
         mlir::Value multiplierConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1000, 64);
         values[0] = rewriter.create<mlir::arith::MulIOp>(loc, values[0], multiplierConst);
      }
      // PGX-LOWER: decimal needs no conversion at the scan boundary. The
      // columnar value already uses the temporary Numeric datum carrier.
      rewriter.replaceOp(atOp, values);
      return success();
   }
};
class AppendTBLowering : public ConversionPattern {
   public:
   explicit AppendTBLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::dsa::Append::getOperationName(), 2, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      mlir::dsa::AppendAdaptor adaptor(operands);
      auto appendOp = mlir::cast<mlir::dsa::Append>(op);
      if (!appendOp.getDs().getType().isa<mlir::dsa::TableBuilderType>()) {
         return mlir::failure();
      }
      auto t = appendOp.getVal().getType();
      if (typeConverter->isLegal(t)) {
         rewriter.modifyOpInPlace(op, [&]() {
            appendOp->setOperands(operands);
         });
         return mlir::success();
      }
      auto* context = getContext();
      mlir::Type arrowPhysicalType = typeConverter->convertType(t);
      if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
          arrowPhysicalType = dateType.getUnit() == mlir::db::DateUnitAttr::day ? mlir::IntegerType::get(context, 32)
                                                                                : mlir::IntegerType::get(context, 64);
      }

      mlir::Value val = adaptor.getVal();
      mlir::Value valid = adaptor.getValid();
      if (t.isa<mlir::db::DateType, mlir::db::TimestampType>()) {
         size_t multiplier = 1;
         if (auto dateType = t.dyn_cast_or_null<mlir::db::DateType>()) {
            multiplier = dateType.getUnit() == mlir::db::DateUnitAttr::day ? 86400000000000 : 1000000;
         } else if (auto timeStampType = t.dyn_cast_or_null<mlir::db::TimestampType>()) {
            multiplier = 1000;  // nanoseconds to microseconds
         }
         if (multiplier != 1) {
            mlir::Value multiplierConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, multiplier, 64);
            val = rewriter.create<mlir::arith::DivSIOp>(loc, val, multiplierConst);
         }
         if (arrowPhysicalType != rewriter.getI64Type()) {
            val = rewriter.create<mlir::arith::TruncIOp>(loc, arrowPhysicalType, val);
         }
      } else if (t.isa<mlir::db::IntervalType>()) {
         mlir::Value multiplierConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1000, 64);
         val = rewriter.create<mlir::arith::DivSIOp>(loc, val, multiplierConst);
      }
      bool numericDatum = isNumericCarrierType(t);
      bool intervalCarrier = mlir::isa<mlir::db::PgIntervalType>(getNonNullablePgBaseType(t));
      if (numericDatum && mlir::isa<mlir::TupleType>(val.getType())) {
          auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, val);
          mlir::Value trueValue = rewriter.create<arith::ConstantOp>(loc,
                                                                     rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
          mlir::Value notNull = rewriter.create<arith::XOrIOp>(loc, unpacked.getVals()[0], trueValue);
          val = unpacked.getVals()[1];
          valid = valid ? rewriter.create<arith::AndIOp>(loc, valid, notNull).getResult() : notNull;
      }
      // PGX-LOWER: decimal needs no conversion at the materialize boundary — the
      // compute value already holds the Numeric datum carrier, matching
      // columnar storage.
      auto newAppend = rewriter.create<mlir::dsa::Append>(loc, adaptor.getDs(), val, valid);
      if (numericDatum) {
          newAppend->setAttr("pgx_numeric_datum", rewriter.getUnitAttr());
      }
      if (intervalCarrier) {
          newAppend->setAttr("pgx_interval_carrier", rewriter.getUnitAttr());
      }

      rewriter.eraseOp(op);
      return success();
   }
};
static mlir::Value numericCarrierToDatum(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value numericCarrier);
static mlir::Value datumToNumericCarrier(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value datumI64);
class StringCastOpLowering : public OpConversionPattern<mlir::db::CastOp> {
   public:
   using OpConversionPattern<mlir::db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::CastOp castOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = castOp->getLoc();
      auto scalarSourceType = castOp.getVal().getType();
      auto scalarTargetType = castOp.getType();
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (!scalarSourceType.isa<mlir::db::StringType>() && !scalarTargetType.isa<mlir::db::StringType>()) return failure();

      Value valueToCast = adaptor.getVal();
      Value result;
      if (scalarSourceType == scalarTargetType) {
         //nothing to do here
      } else if (auto stringType = scalarSourceType.dyn_cast_or_null<db::StringType>()) {
         if (auto intWidth = getIntegerWidth(scalarTargetType, false)) {
            result = rt::StringRuntime::toInt(rewriter, loc)({valueToCast})[0];
            if (intWidth < 64) {
               result = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, result);
            }
         } else if (auto floatType = scalarTargetType.dyn_cast_or_null<FloatType>()) {
            result = floatType.getWidth() == 32 ? rt::StringRuntime::toFloat32(rewriter, loc)({valueToCast})[0] : rt::StringRuntime::toFloat64(rewriter, loc)({valueToCast})[0];
         } else if (auto decimalType = scalarTargetType.dyn_cast_or_null<db::DecimalType>()) {
             Value datum = rt::NumericRuntime::pgx_numeric_from_string(rewriter, loc)({valueToCast})[0];
             result = datumToNumericCarrier(rewriter, loc, datum);
         }
      } else if (auto intWidth = getIntegerWidth(scalarSourceType, false)) {
         result = rt::StringRuntime::fromInt(rewriter, loc)({valueToCast})[0];
      } else if (auto floatType = scalarSourceType.dyn_cast_or_null<FloatType>()) {
         result = floatType.getWidth() == 32 ? rt::StringRuntime::fromFloat32(rewriter, loc)({valueToCast})[0] : rt::StringRuntime::fromFloat64(rewriter, loc)({valueToCast})[0];
      } else if (auto decimalSourceType = scalarSourceType.dyn_cast_or_null<db::DecimalType>()) {
          Value datum = numericCarrierToDatum(rewriter, loc, valueToCast);
          result = rt::NumericRuntime::pgx_numeric_to_string(rewriter, loc)({datum})[0];
      } else if (auto charType = scalarSourceType.dyn_cast_or_null<db::CharType>()) {
         auto bytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(charType.getBytes()));
         result = rt::StringRuntime::fromChar(rewriter, loc)({valueToCast, bytes})[0];
      }
      if (result) {
         rewriter.replaceOp(castOp, result);
         return success();
      } else {
         return failure();
      }
   }
};
class StringCmpOpLowering : public OpConversionPattern<mlir::db::CmpOp> {
   public:
   using OpConversionPattern<mlir::db::CmpOp>::OpConversionPattern;

   bool stringIsOk(std::string str) const {
      for (auto x : str) {
         if (!std::isalnum(x)) return false;
      }
      return true;
   }
   LogicalResult
   matchAndRewrite(mlir::db::CmpOp cmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
       auto type = getBaseType(cmpOp.getLeft().getType());
       if (!type.isa<db::StringType>()
           && !mlir::isa<mlir::db::PgTextType, mlir::db::PgVarcharType, mlir::db::PgBpcharType>(type))
       {
           return failure();
       }
       Value res;
       auto leftOperand = unwrapNullableOperand(rewriter, cmpOp->getLoc(), adaptor.getLeft());
       auto rightOperand = unwrapNullableOperand(rewriter, cmpOp->getLoc(), adaptor.getRight());
       Value left = leftOperand.payload;
       Value right = rightOperand.payload;
       const uint32_t pgFunctionOid = pgStringCompareFunctionOid(type, cmpOp.getPredicate());
       if (mlir::db::isPgValueType(type) && pgFunctionOid != InvalidOid) {
           Value leftTypeOid = rewriter.create<arith::ConstantIntOp>(cmpOp->getLoc(), mlir::db::getPgTypeOid(type), 32);
           Value rightTypeOid = rewriter.create<arith::ConstantIntOp>(cmpOp->getLoc(), mlir::db::getPgTypeOid(type), 32);
           Value functionOid = rewriter.create<arith::ConstantIntOp>(cmpOp->getLoc(), pgFunctionOid, 32);
           Value collationOid = rewriter.create<arith::ConstantIntOp>(cmpOp->getLoc(), mlir::db::getPgCollation(type),
                                                                      32);
           res = rt::StringRuntime::pgCallBool2(
               rewriter, cmpOp->getLoc())({left, leftTypeOid, right, rightTypeOid, functionOid, collationOid})[0];
       } else {
           switch (cmpOp.getPredicate()) {
           case db::DBCmpPredicate::eq:
               res = rt::StringRuntime::compareEq(rewriter, cmpOp->getLoc())({left, right})[0];
               break;
           case db::DBCmpPredicate::neq:
               res = rt::StringRuntime::compareNEq(rewriter, cmpOp->getLoc())({left, right})[0];
               break;
           case db::DBCmpPredicate::lt:
               res = rt::StringRuntime::compareLt(rewriter, cmpOp->getLoc())({left, right})[0];
               break;
           case db::DBCmpPredicate::gt:
               res = rt::StringRuntime::compareGt(rewriter, cmpOp->getLoc())({left, right})[0];
               break;
           case db::DBCmpPredicate::lte:
               res = rt::StringRuntime::compareLte(rewriter, cmpOp->getLoc())({left, right})[0];
               break;
           case db::DBCmpPredicate::gte:
               res = rt::StringRuntime::compareGte(rewriter, cmpOp->getLoc())({left, right})[0];
               break;
           }
       }
       if (mlir::Value isNull = combineNullFlags(rewriter, cmpOp->getLoc(), leftOperand.isNull, rightOperand.isNull)) {
           mlir::Type convertedResultType = typeConverter->convertType(cmpOp.getType());
           if (mlir::isa<mlir::TupleType>(convertedResultType)) {
               rewriter.replaceOpWithNewOp<mlir::util::PackOp>(cmpOp, convertedResultType, mlir::ValueRange{isNull, res});
               return success();
           }
           mlir::Value falseValue = rewriter.create<arith::ConstantOp>(
               cmpOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
           res = rewriter.create<arith::SelectOp>(cmpOp->getLoc(), isNull, falseValue, res);
       }
       rewriter.replaceOp(cmpOp, res);
       return success();
   }
};

class RuntimeCallLowering : public OpConversionPattern<mlir::db::RuntimeCall> {
   public:
   using OpConversionPattern<mlir::db::RuntimeCall>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::RuntimeCall runtimeCallOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
      auto* fn = reg->lookup(runtimeCallOp.getFn().str());
      if (!fn) return failure();
      Value result;
      mlir::Type resType = runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : mlir::Type();
      llvm::SmallVector<Value> callArgs;
      callArgs.reserve(adaptor.getArgs().size());
      Value isNull;
      const bool needsWrapping = fn->nullHandleType == mlir::db::RuntimeFunction::NeedsWrapping;
      for (auto [originalArg, loweredArg] : llvm::zip(runtimeCallOp.getArgs(), adaptor.getArgs())) {
          if (needsWrapping && isNullableDbValueType(originalArg.getType())) {
              auto unwrapped = unwrapNullableOperand(rewriter, runtimeCallOp->getLoc(), loweredArg);
              isNull = combineNullFlags(rewriter, runtimeCallOp->getLoc(), isNull, unwrapped.isNull);
              callArgs.push_back(unwrapped.payload);
          } else {
              callArgs.push_back(loweredArg);
          }
      }
      if (std::holds_alternative<mlir::util::FunctionSpec>(fn->implementation)) {
         auto& implFn = std::get<mlir::util::FunctionSpec>(fn->implementation);
         auto resRange = implFn(rewriter, rewriter.getUnknownLoc())(callArgs);
         assert((resRange.size() == 1 && resType) || (resRange.empty() && !resType));
         result = resRange.size() == 1 ? resRange[0] : mlir::Value();
      } else if (std::holds_alternative<mlir::db::RuntimeFunction::loweringFnT>(fn->implementation)) {
         auto& implFn = std::get<mlir::db::RuntimeFunction::loweringFnT>(fn->implementation);
         result = implFn(rewriter, callArgs, runtimeCallOp.getArgs().getTypes(),
                         runtimeCallOp->getNumResults() == 1 ? runtimeCallOp->getResultTypes()[0] : ::mlir::Type(),
                         const_cast<mlir::TypeConverter*>(typeConverter), runtimeCallOp->getLoc());
      }

      if (runtimeCallOp->getNumResults() == 0) {
         rewriter.eraseOp(runtimeCallOp);
      } else {
          if (needsWrapping && isNull && isNullableDbValueType(resType)) {
              mlir::Type convertedResultType = typeConverter->convertType(resType);
              result = rewriter.create<mlir::util::PackOp>(runtimeCallOp->getLoc(), convertedResultType,
                                                           mlir::ValueRange{isNull, result});
          }
         rewriter.replaceOp(runtimeCallOp, result);
      }
      return success();
   }
};

class NotOpLowering : public OpConversionPattern<mlir::db::NotOp> {
   public:
   using OpConversionPattern<mlir::db::NotOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::NotOp notOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
       auto loc = notOp->getLoc();
       Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
       if (isNullableDbValueType(notOp.getVal().getType())) {
           auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.getVal());
           Value negated = rewriter.create<arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, unPackOp.getVals()[1],
                                                          falseValue);
           Value combined = rewriter.create<mlir::util::PackOp>(loc, ValueRange({unPackOp.getVals()[0], negated}));
           rewriter.replaceOp(notOp, combined);
           return success();
       }
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(notOp, mlir::arith::CmpIPredicate::eq, adaptor.getVal(), falseValue);
      return success();
   }
};
class DeriveTruthLowering : public OpConversionPattern<mlir::db::DeriveTruth> {
   public:
    using OpConversionPattern<mlir::db::DeriveTruth>::OpConversionPattern;
    LogicalResult matchAndRewrite(mlir::db::DeriveTruth deriveTruthOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto originalType = deriveTruthOp.getVal().getType();
        if (isNullableDbValueType(originalType)) {
            auto unPackOp = rewriter.create<mlir::util::UnPackOp>(deriveTruthOp->getLoc(), adaptor.getVal());
            auto trueValue = rewriter.create<arith::ConstantOp>(deriveTruthOp->getLoc(),
                                                                rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
            auto notNull = rewriter.create<arith::XOrIOp>(deriveTruthOp->getLoc(), unPackOp.getVals()[0], trueValue);
            rewriter.replaceOpWithNewOp<arith::AndIOp>(deriveTruthOp, notNull, unPackOp.getVals()[1]);
            return success();
        }

        if (!adaptor.getVal().getType().isInteger(1)) {
            return failure();
        }
        rewriter.replaceOp(deriveTruthOp, adaptor.getVal());
        return success();
    }
};
class AndOpLowering : public OpConversionPattern<mlir::db::AndOp> {
   public:
   using OpConversionPattern<mlir::db::AndOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::AndOp andOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value result;
      Value isNull;
      auto loc = andOp->getLoc();

      for (size_t i = 0; i < adaptor.getVals().size(); i++) {
         auto currType = andOp.getVals()[i].getType();
         bool currNullable = isNullableDbValueType(currType);
         Value currNull;
         Value currVal;
         if (currNullable) {
            auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.getVals()[i]);
            currNull = unPackOp.getVals()[0];
            currVal = unPackOp.getVals()[1];
         } else {
            currVal = adaptor.getVals()[i];
         }
         if (i == 0) {
            if (currNullable) {
               result = rewriter.create<arith::OrIOp>(loc, currNull, currVal);
            } else {
               result = currVal;
            }
            isNull = currNull;
         } else {
            if (currNullable) {
               if (isNull) {
                  isNull = rewriter.create<arith::OrIOp>(loc, isNull, currNull);
               } else {
                  isNull = currNull;
               }
            }
            if (currNullable) {
               result = rewriter.create<arith::SelectOp>(loc, currNull, result, rewriter.create<arith::AndIOp>(loc, currVal, result));
            } else {
               result = rewriter.create<arith::AndIOp>(loc, currVal, result);
            }
         }
      }
      if (isNullableDbValueType(andOp.getResult().getType())) {
          isNull = rewriter.create<arith::AndIOp>(loc, result, isNull);
          Value combined = rewriter.create<mlir::util::PackOp>(loc, ValueRange({isNull, result}));
          rewriter.replaceOp(andOp, combined);
      } else {
          rewriter.replaceOp(andOp, result);
      }
      return success();
   }
};
class OrOpLowering : public OpConversionPattern<mlir::db::OrOp> {
   public:
   using OpConversionPattern<mlir::db::OrOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::OrOp orOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      Value result;
      Value isNull;
      auto loc = orOp->getLoc();
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      for (size_t i = 0; i < adaptor.getVals().size(); i++) {
         auto currType = orOp.getVals()[i].getType();
         bool currNullable = isNullableDbValueType(currType);
         Value currNull;
         Value currVal;
         if (currNullable) {
             auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, adaptor.getVals()[i]);
             currNull = unPackOp.getVals()[0];
             currVal = unPackOp.getVals()[1];
         } else {
             currVal = adaptor.getVals()[i];
         }
         if (i == 0) {
             if (currNullable) {
                 result = rewriter.create<arith::SelectOp>(loc, currNull, falseValue, currVal);
             } else {
                 result = currVal;
             }
             isNull = currNull;
         } else {
             if (currNullable) {
                 if (isNull) {
                     isNull = rewriter.create<arith::OrIOp>(loc, isNull, currNull);
                 } else {
                     isNull = currNull;
                 }
             }
             if (currNullable) {
                 result = rewriter.create<arith::SelectOp>(loc, currNull, result,
                                                           rewriter.create<arith::OrIOp>(loc, currVal, result));
             } else {
                 result = rewriter.create<arith::OrIOp>(loc, currVal, result);
             }
         }
      }
      if (isNullableDbValueType(orOp.getResult().getType())) {
          isNull = rewriter.create<arith::SelectOp>(loc, result, falseValue, isNull);
          Value combined = rewriter.create<mlir::util::PackOp>(loc, ValueRange({isNull, result}));
          rewriter.replaceOp(orOp, combined);
      } else {
          rewriter.replaceOp(orOp, result);
      }
      return success();
   }
};

template <class OpClass, class OperandType, class StdOpClass>
class BinOpLowering : public ConversionPattern {
   public:
   explicit BinOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, OpClass::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto binOp = mlir::cast<OpClass>(op);
      typename OpClass::Adaptor adaptor(operands);
      auto type = getBaseType(binOp.getLeft().getType());
      if (!supportsPhysicalBinOp<OperandType>(type)) {
          return failure();
      }

      auto loc = binOp->getLoc();
      auto leftOperand = unwrapNullableOperand(rewriter, loc, adaptor.getLeft());
      auto rightOperand = unwrapNullableOperand(rewriter, loc, adaptor.getRight());
      if (!leftOperand.payload.getType().template isa<OperandType>()
          || !rightOperand.payload.getType().template isa<OperandType>())
      {
          return failure();
      }

      int64_t rightNullFallback = mlir::isa<mlir::db::DivOp, mlir::db::ModOp>(op) ? 1 : 0;
      mlir::Value leftPayload = safePayloadOr(rewriter, loc, leftOperand.payload, leftOperand.isNull, 0);
      mlir::Value rightPayload = safePayloadOr(rewriter, loc, rightOperand.payload, rightOperand.isNull,
                                               rightNullFallback);
      mlir::Value result = rewriter.template create<StdOpClass>(loc, leftPayload, rightPayload);
      mlir::Type convertedResultType = this->typeConverter->convertType(binOp.getResult().getType());
      mlir::Type resultPayloadType = convertedResultType;
      if (auto tupleType = mlir::dyn_cast_or_null<mlir::TupleType>(convertedResultType)) {
          resultPayloadType = tupleType.getType(1);
      }
      result = castPhysicalScalar(rewriter, loc, result, resultPayloadType);
      if (mlir::Value isNull = combineNullFlags(rewriter, loc, leftOperand.isNull, rightOperand.isNull)) {
          if (mlir::isa<mlir::TupleType>(convertedResultType)) {
              rewriter.template replaceOpWithNewOp<mlir::util::PackOp>(binOp, convertedResultType,
                                                                       mlir::ValueRange{isNull, result});
              return success();
          }
      }
      rewriter.replaceOp(binOp, result);
      return success();
   }
};
// PGX-LOWER: DecimalType lowers to a Datum-width Numeric datum carrier. These
// helpers keep the call sites explicit while Milestone 1/2 code paths converge.
static mlir::Value numericCarrierToDatum(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value numericCarrier) {
    if (numericCarrier.getType() == builder.getI64Type()) {
        return numericCarrier;
    }
    return builder.create<mlir::arith::TruncIOp>(loc, builder.getI64Type(), numericCarrier);
}
static mlir::Value datumToNumericCarrier(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value datumI64) {
    if (datumI64.getType() == builder.getI64Type()) {
        return datumI64;
    }
    return builder.create<mlir::arith::ExtUIOp>(loc, builder.getIntegerType(128), datumI64);
}
static bool isNumericCarrierType(mlir::Type type) {
    type = getBaseType(type);
    if (mlir::db::isPgValueType(type)) {
        type = mlir::db::withPgNullability(type, mlir::db::PgNullability::Never);
    }
    return mlir::isa<mlir::db::DecimalType, mlir::db::PgNumericType>(type);
}
struct NumericOperand {
    mlir::Value payload;
    mlir::Value isNull;
};
static NumericOperand unwrapNumericOperand(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value) {
    mlir::Value isNull;
    while (auto tupleType = mlir::dyn_cast_or_null<mlir::TupleType>(value.getType())) {
        if (tupleType.size() != 2 || !tupleType.getType(0).isInteger(1)) {
            break;
        }
        auto unpacked = builder.create<mlir::util::UnPackOp>(loc, value);
        isNull = combineNullFlags(builder, loc, isNull, unpacked.getVals()[0]);
        value = unpacked.getVals()[1];
    }
    return {value, isNull};
}
static mlir::Value
combineNullFlags(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value leftNull, mlir::Value rightNull) {
    if (leftNull && rightNull) {
        return builder.create<mlir::arith::OrIOp>(loc, leftNull, rightNull);
    }
    if (leftNull) {
        return leftNull;
    }
    return rightNull;
}
static mlir::Value numericIntCarrier(mlir::OpBuilder& builder, mlir::Location loc, int64_t value) {
    mlir::Value integer = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(value));
    mlir::Value datum = rt::NumericRuntime::pgx_int_to_numeric(builder, loc)({integer})[0];
    return datumToNumericCarrier(builder, loc, datum);
}
static mlir::Value
safeNumericPayload(mlir::OpBuilder& builder, mlir::Location loc, const NumericOperand& operand, mlir::Value fallback) {
    if (!operand.isNull) {
        return operand.payload;
    }
    return builder.create<mlir::arith::SelectOp>(loc, operand.isNull, fallback, operand.payload);
}
template <class DBOp, class Op>
class DecimalOpScaledLowering : public ConversionPattern {
   public:
   explicit DecimalOpScaledLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, DBOp::getOperationName(), 1, context) {}
   
   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto decimalOp = mlir::cast<DBOp>(op);
      typename DBOp::Adaptor adaptor(operands);
      auto type = getBaseType(decimalOp.getType());
      if (isNumericCarrierType(type)) {
          // PGX-LOWER: decimal div/mod via PG-native numeric_div/numeric_mod.
          // Operands are Numeric datum carriers; extract, call, rewrap.
          auto loc = decimalOp->getLoc();
          auto leftOperand = unwrapNumericOperand(rewriter, loc, adaptor.getLeft());
          auto rightOperand = unwrapNumericOperand(rewriter, loc, adaptor.getRight());
          mlir::Value zero = numericIntCarrier(rewriter, loc, 0);
          mlir::Value one = numericIntCarrier(rewriter, loc, 1);
          mlir::Value leftPayload = safeNumericPayload(rewriter, loc, leftOperand, zero);
          mlir::Value rightPayload = safeNumericPayload(rewriter, loc, rightOperand, one);
          mlir::Value left = numericCarrierToDatum(rewriter, loc, leftPayload);
          mlir::Value right = numericCarrierToDatum(rewriter, loc, rightPayload);
          mlir::Value result;
          if (mlir::isa<mlir::db::DivOp>(op)) {
              result = rt::NumericRuntime::pgx_numeric_div(rewriter, loc)({left, right})[0];
          } else if (mlir::isa<mlir::db::ModOp>(op)) {
              result = rt::NumericRuntime::pgx_numeric_mod(rewriter, loc)({left, right})[0];
          } else {
              return failure();
          }
          mlir::Value resultValue = datumToNumericCarrier(rewriter, loc, result);
          if (mlir::Value isNull = combineNullFlags(rewriter, loc, leftOperand.isNull, rightOperand.isNull)) {
              resultValue = rewriter.create<mlir::util::PackOp>(
                  loc, this->typeConverter->convertType(decimalOp.getType()), mlir::ValueRange{isNull, resultValue});
          }
          rewriter.replaceOp(decimalOp, resultValue);
          return success();
      }
      return failure();
   }
};
template <class DBOp, class ArithOp>
class DecimalBinOpLowering : public ConversionPattern {
   public:
   explicit DecimalBinOpLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, DBOp::getOperationName(), 1, context) {}

   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      auto binOp = mlir::cast<DBOp>(op);
      typename DBOp::Adaptor adaptor(operands);
      if (isNumericCarrierType(binOp.getType())) {
          // PGX-LOWER: operands are Numeric datum carriers. Extract the
          // datums, call the PG-native numeric_* runtime fn, rewrap the result.
          // PG handles scale, so the old scale-multiply/divide adjustment is gone.
          auto loc = binOp->getLoc();
          auto leftOperand = unwrapNumericOperand(rewriter, loc, adaptor.getLeft());
          auto rightOperand = unwrapNumericOperand(rewriter, loc, adaptor.getRight());
          mlir::Value zero = numericIntCarrier(rewriter, loc, 0);
          mlir::Value leftPayload = safeNumericPayload(rewriter, loc, leftOperand, zero);
          mlir::Value rightPayload = safeNumericPayload(rewriter, loc, rightOperand, zero);
          mlir::Value left = numericCarrierToDatum(rewriter, loc, leftPayload);
          mlir::Value right = numericCarrierToDatum(rewriter, loc, rightPayload);
          mlir::Value result;
          if (mlir::isa<mlir::db::AddOp>(op)) {
              result = rt::NumericRuntime::pgx_numeric_add(rewriter, loc)({left, right})[0];
          } else if (mlir::isa<mlir::db::SubOp>(op)) {
              result = rt::NumericRuntime::pgx_numeric_sub(rewriter, loc)({left, right})[0];
          } else if (mlir::isa<mlir::db::MulOp>(op)) {
              result = rt::NumericRuntime::pgx_numeric_mul(rewriter, loc)({left, right})[0];
          } else {
              return failure();
          }
          mlir::Value resultValue = datumToNumericCarrier(rewriter, loc, result);
          if (mlir::Value isNull = combineNullFlags(rewriter, loc, leftOperand.isNull, rightOperand.isNull)) {
              resultValue = rewriter.create<mlir::util::PackOp>(loc, this->typeConverter->convertType(binOp.getType()),
                                                                mlir::ValueRange{isNull, resultValue});
          }
          rewriter.replaceOp(binOp, resultValue);
          return success();
      }
      return failure();
   }
};

// Lowering for date/interval arithmetic with potentially mixed operand types
template <class OpClass, class StdOpClass>
class DateIntervalArithmeticLowering : public OpConversionPattern<OpClass> {
   public:
   using OpConversionPattern<OpClass>::OpConversionPattern;
   using OpAdaptor = typename OpClass::Adaptor;

   static mlir::Value pgIntervalToMicroseconds(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value interval) {
       auto unpacked = builder.create<mlir::util::UnPackOp>(loc, interval);
       mlir::Value time = unpacked.getVals()[0];
       mlir::Value day = unpacked.getVals()[1];
       if (day.getType().isInteger(32)) {
           day = builder.create<mlir::arith::ExtSIOp>(loc, builder.getI64Type(), day);
       }
       mlir::Value multiplier = builder.create<mlir::arith::ConstantIntOp>(loc, kPgMicrosecondsPerDay, 64);
       mlir::Value dayMicroseconds = builder.create<mlir::arith::MulIOp>(loc, day, multiplier);
       return builder.create<mlir::arith::AddIOp>(loc, time, dayMicroseconds);
   }

   LogicalResult matchAndRewrite(OpClass binOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto leftType = getBaseType(binOp.getLeft().getType());
      auto rightType = getBaseType(binOp.getRight().getType());

      const bool leftIsPgDate = isPgDateValueType(leftType);
      const bool rightIsPgDate = isPgDateValueType(rightType);
      const bool leftIsPgInt4 = isPgInt4ValueType(leftType);
      const bool rightIsPgInt4 = isPgInt4ValueType(rightType);
      const bool resultIsPgDate = isPgDateValueType(binOp.getType());
      const bool resultIsPgInt4 = isPgInt4ValueType(binOp.getType());
      const bool leftIsPgInterval = mlir::isa<mlir::db::PgIntervalType>(getNonNullablePgBaseType(leftType));
      const bool rightIsPgInterval = mlir::isa<mlir::db::PgIntervalType>(getNonNullablePgBaseType(rightType));

      const bool dateInt4Add = std::is_same_v<OpClass, mlir::db::AddOp> && resultIsPgDate
                               && ((leftIsPgDate && rightIsPgInt4) || (leftIsPgInt4 && rightIsPgDate));
      const bool dateInt4Sub = std::is_same_v<OpClass, mlir::db::SubOp> && resultIsPgDate && leftIsPgDate
                               && rightIsPgInt4;
      const bool dateDateSub = std::is_same_v<OpClass, mlir::db::SubOp> && resultIsPgInt4 && leftIsPgDate
                               && rightIsPgDate;
      if (dateInt4Add || dateInt4Sub || dateDateSub) {
          auto loc = binOp->getLoc();
          auto leftOperand = unwrapNullableOperand(rewriter, loc, adaptor.getLeft());
          auto rightOperand = unwrapNullableOperand(rewriter, loc, adaptor.getRight());

          if (!leftOperand.payload.getType().isInteger(32) || !rightOperand.payload.getType().isInteger(32)) {
              return failure();
          }

          mlir::Value leftPayload = safePayloadOr(rewriter, loc, leftOperand.payload, leftOperand.isNull, 0);
          mlir::Value rightPayload = safePayloadOr(rewriter, loc, rightOperand.payload, rightOperand.isNull, 0);
          mlir::Value result;
          if constexpr (std::is_same_v<OpClass, mlir::db::AddOp>) {
              result = rewriter.create<mlir::arith::AddIOp>(loc, leftPayload, rightPayload);
          } else {
              result = rewriter.create<mlir::arith::SubIOp>(loc, leftPayload, rightPayload);
          }

          mlir::Type convertedResultType = this->typeConverter->convertType(binOp.getType());
          mlir::Type resultPayloadType = convertedResultType;
          if (auto tupleType = mlir::dyn_cast_or_null<mlir::TupleType>(convertedResultType)) {
              resultPayloadType = tupleType.getType(1);
          }
          result = castPhysicalScalar(rewriter, loc, result, resultPayloadType);

          if (mlir::Value isNull = combineNullFlags(rewriter, loc, leftOperand.isNull, rightOperand.isNull)) {
              if (mlir::isa<mlir::TupleType>(convertedResultType)) {
                  rewriter.replaceOpWithNewOp<mlir::util::PackOp>(binOp, convertedResultType,
                                                                  mlir::ValueRange{isNull, result});
                  return success();
              }
          }
          rewriter.replaceOp(binOp, result);
          return success();
      }

      if ((leftIsPgDate && rightIsPgInterval) || (leftIsPgInterval && rightIsPgDate)) {
          auto loc = binOp->getLoc();
          auto leftOperand = unwrapNullableOperand(rewriter, loc, adaptor.getLeft());
          auto rightOperand = unwrapNullableOperand(rewriter, loc, adaptor.getRight());

          mlir::Value dateMicroseconds;
          mlir::Value intervalMicroseconds;
          if (leftIsPgDate && rightIsPgInterval) {
              if (!mlir::isa<mlir::TupleType>(rightOperand.payload.getType())) {
                  return failure();
              }
              dateMicroseconds = pgDateDaysToTimestampMicros(rewriter, loc, leftOperand.payload);
              intervalMicroseconds = pgIntervalToMicroseconds(rewriter, loc, rightOperand.payload);
          } else {
              if (!mlir::isa<mlir::TupleType>(leftOperand.payload.getType())) {
                  return failure();
              }
              dateMicroseconds = pgDateDaysToTimestampMicros(rewriter, loc, rightOperand.payload);
              intervalMicroseconds = pgIntervalToMicroseconds(rewriter, loc, leftOperand.payload);
          }

          mlir::Value result;
          if constexpr (std::is_same_v<OpClass, mlir::db::AddOp>) {
              result = rewriter.create<mlir::arith::AddIOp>(loc, dateMicroseconds, intervalMicroseconds);
          } else {
              if (!leftIsPgDate || !rightIsPgInterval) {
                  return failure();
              }
              result = rewriter.create<mlir::arith::SubIOp>(loc, dateMicroseconds, intervalMicroseconds);
          }

          if (mlir::Value isNull = combineNullFlags(rewriter, loc, leftOperand.isNull, rightOperand.isNull)) {
              auto convertedResultType = this->typeConverter->convertType(binOp.getType());
              if (mlir::isa<mlir::TupleType>(convertedResultType)) {
                  rewriter.replaceOpWithNewOp<mlir::util::PackOp>(binOp, convertedResultType,
                                                                  mlir::ValueRange{isNull, result});
                  return success();
              }
          }
          rewriter.replaceOp(binOp, result);
          return success();
      }

      // Match if either operand is DateType or IntervalType
      bool leftIsDateTime = mlir::isa<mlir::db::DateType, mlir::db::IntervalType>(leftType);
      bool rightIsDateTime = mlir::isa<mlir::db::DateType, mlir::db::IntervalType>(rightType);

      if (!leftIsDateTime && !rightIsDateTime) {
         return failure();  // Not a date/interval operation
      }

      PGX_LOG(DB_LOWER, DEBUG, "[DateIntervalArithmeticLowering] Matched date/interval operation");

      // DateType and IntervalType both convert to i64
      auto i64Type = rewriter.getI64Type();
      mlir::Value left = adaptor.getLeft();
      mlir::Value right = adaptor.getRight();

      // Ensure operands are i64 (extend from i32 if needed)
      if (left.getType() != i64Type) {
         if (left.getType().isInteger(32)) {
            PGX_LOG(DB_LOWER, DEBUG, "[DateIntervalArithmeticLowering] Extending left operand from i32 to i64");
            left = rewriter.create<arith::ExtSIOp>(binOp->getLoc(), i64Type, left);
         } else {
            return failure();
         }
      }
      if (right.getType() != i64Type) {
         if (right.getType().isInteger(32)) {
            PGX_LOG(DB_LOWER, DEBUG, "[DateIntervalArithmeticLowering] Extending right operand from i32 to i64");
            right = rewriter.create<arith::ExtSIOp>(binOp->getLoc(), i64Type, right);
         } else {
            return failure();
         }
      }

      // Create the arithmetic operation
      rewriter.replaceOpWithNewOp<StdOpClass>(binOp, i64Type, left, right);
      return success();
   }
};

class IsNullOpLowering : public OpConversionPattern<mlir::db::IsNullOp> {
   public:
   using OpConversionPattern<mlir::db::IsNullOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::IsNullOp isNullOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
       auto originalType = isNullOp.getVal().getType();
       if (originalType.isa<mlir::db::NullableType>()
           || (mlir::db::isPgValueType(originalType)
               && mlir::db::getPgNullability(originalType) == mlir::db::PgNullability::Maybe))
       {
           auto unPackOp = rewriter.create<mlir::util::UnPackOp>(isNullOp->getLoc(), adaptor.getVal());
           rewriter.replaceOp(isNullOp, unPackOp.getVals()[0]);
       } else if (mlir::db::isPgValueType(originalType)) {
           rewriter.replaceOpWithNewOp<arith::ConstantOp>(isNullOp, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
       } else {
           rewriter.replaceOp(isNullOp, adaptor.getVal());
       }
      return success();
   }
};
class NullableGetValOpLowering : public OpConversionPattern<mlir::db::NullableGetVal> {
   public:
   using OpConversionPattern<mlir::db::NullableGetVal>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::NullableGetVal op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto unPackOp = rewriter.create<mlir::util::UnPackOp>(op->getLoc(), adaptor.getVal());
      rewriter.replaceOp(op, unPackOp.getVals()[1]);
      return success();
   }
};
class AsNullableOpLowering : public OpConversionPattern<mlir::db::AsNullableOp> {
   public:
   using OpConversionPattern<mlir::db::AsNullableOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::AsNullableOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      mlir::Value isNull = adaptor.getNull();
      if (!isNull) {
         isNull = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      }
      auto packOp = rewriter.create<mlir::util::PackOp>(op->getLoc(), ValueRange({isNull, adaptor.getVal()}));
      rewriter.replaceOp(op, packOp.getTuple());
      return success();
   }
};
class NullOpLowering : public OpConversionPattern<mlir::db::NullOp> {
   public:
   using OpConversionPattern<mlir::db::NullOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::NullOp nullOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
       auto tupleType = typeConverter->convertType(nullOp.getType()).dyn_cast_or_null<mlir::TupleType>();
       if (!tupleType) {
           return failure();
       }
      auto undefValue = rewriter.create<mlir::util::UndefOp>(nullOp->getLoc(), tupleType.getType(1));
      auto trueValue = rewriter.create<arith::ConstantOp>(nullOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      rewriter.replaceOpWithNewOp<mlir::util::PackOp>(nullOp, tupleType, ValueRange({trueValue, undefValue}));
      return success();
   }
};

class ConstantLowering : public OpConversionPattern<mlir::db::ConstantOp> {
   static std::tuple<int, uint32_t, uint32_t> convertTypeToArrow(::mlir::Type type) {
      int typeConstant = 0;  // Use PostgreSQL OIDs instead of enum
      uint32_t param1 = 0, param2 = 0;
      if (mlir::db::isPgValueType(type)) {
          if (mlir::isa<mlir::db::PgTextType, mlir::db::PgVarcharType, mlir::db::PgBpcharType>(type)) {
              typeConstant = TEXTOID;
          } else if (mlir::isa<mlir::db::PgTimestampType>(type)) {
              typeConstant = TIMESTAMPOID;
              param1 = static_cast<uint32_t>(support::MICRO);
          } else {
              typeConstant = mlir::db::getPgTypeOid(type);
          }
      } else if (isIntegerType(type, 1)) {
          typeConstant = BOOLOID;
      } else if (auto intWidth = getIntegerWidth(type, false)) {
          switch (intWidth) {
          case 8: typeConstant = INT2OID; break;
          case 16: typeConstant = INT2OID; break;
          case 32: typeConstant = INT4OID; break;
          case 64: typeConstant = INT8OID; break;
          }
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
          // PostgreSQL doesn't have unsigned types, map to signed equivalents
          switch (uIntWidth) {
          case 8: typeConstant = INT2OID; break; // Map to INT2
          case 16: typeConstant = INT2OID; break; // Map to INT2
          case 32: typeConstant = INT4OID; break; // Map to INT4
          case 64: typeConstant = INT8OID; break; // Map to INT8
          }
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
          typeConstant = NUMERICOID;
          param1 = decimalType.getP();
          param2 = decimalType.getS();
      } else if (auto floatType = type.dyn_cast_or_null<::mlir::FloatType>()) {
          switch (floatType.getWidth()) {
          case 16: typeConstant = FLOAT4OID; break; // Map half to float4
          case 32: typeConstant = FLOAT4OID; break;
          case 64: typeConstant = FLOAT8OID; break;
          }
      } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
          typeConstant = TEXTOID;
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
          typeConstant = DATEOID;
      } else if (auto charType = type.dyn_cast_or_null<mlir::db::CharType>()) {
          typeConstant = TEXTOID;
          param1 = charType.getBytes();
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::db::IntervalType>()) {
          typeConstant = INTERVALOID;
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
          typeConstant = TIMESTAMPOID;
          param1 = static_cast<uint32_t>(timestampType.getUnit());
      }
      // Note: typeConstant will be 0 for unsupported types, handled by caller
      return {typeConstant, param1, param2};
   }

   public:
   using OpConversionPattern<mlir::db::ConstantOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::ConstantOp constantOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto type = constantOp.getType();
      auto stdType = typeConverter->convertType(type);
      if (mlir::isa<mlir::db::PgIntervalType>(type)) {
          auto tupleType = stdType.dyn_cast_or_null<mlir::TupleType>();
          auto fields = constantOp.getConstantValue().dyn_cast_or_null<ArrayAttr>();
          if (!tupleType || !fields || fields.size() != 3) {
              return failure();
          }
          auto time = mlir::cast<IntegerAttr>(fields[0]).getInt();
          auto day = mlir::cast<IntegerAttr>(fields[1]).getInt();
          auto month = mlir::cast<IntegerAttr>(fields[2]).getInt();
          mlir::Value timeValue = rewriter.create<arith::ConstantOp>(
              constantOp->getLoc(), tupleType.getType(0), rewriter.getIntegerAttr(tupleType.getType(0), time));
          mlir::Value dayValue = rewriter.create<arith::ConstantOp>(constantOp->getLoc(), tupleType.getType(1),
                                                                    rewriter.getIntegerAttr(tupleType.getType(1), day));
          mlir::Value monthValue = rewriter.create<arith::ConstantOp>(
              constantOp->getLoc(), tupleType.getType(2), rewriter.getIntegerAttr(tupleType.getType(2), month));
          rewriter.replaceOpWithNewOp<mlir::util::PackOp>(constantOp, tupleType,
                                                          mlir::ValueRange{timeValue, dayValue, monthValue});
          return success();
      }
      auto [arrowType, param1, param2] = convertTypeToArrow(type);

      if (arrowType == 0) {
         return rewriter.notifyMatchFailure(constantOp, "Unsupported type for constant conversion");
      }

      std::stringstream debugMsg;
      debugMsg << "ConstantOp lowering - type OID: " << arrowType << ", param1: " << param1 << ", param2: " << param2;
      PGX_LOG(DB_LOWER, DEBUG, "[DB] %s", debugMsg.str().c_str());

      std::variant<int64_t, double, std::string> parseArg;
      if (auto integerAttr = constantOp.getConstantValue().dyn_cast_or_null<IntegerAttr>()) {
         parseArg = integerAttr.getInt();
      } else if (auto floatAttr = constantOp.getConstantValue().dyn_cast_or_null<FloatAttr>()) {
         parseArg = floatAttr.getValueAsDouble();
      } else if (auto stringAttr = constantOp.getConstantValue().dyn_cast_or_null<StringAttr>()) {
         parseArg = stringAttr.str();
      } else {
         return failure();
      }
      auto parseResult = support::parse(parseArg, arrowType, param1, param2);  // arrowType is already an int (OID)
      if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
          // PGX-LOWER: a decimal literal is a PG Numeric datum. Parse the exact
          // literal text at runtime via numeric_in so wide values, arbitrary
          // scale, NaN, and +/-Infinity follow stock PostgreSQL semantics.
          auto loc = constantOp->getLoc();
          mlir::Value literal = rewriter.create<mlir::util::CreateConstVarLen>(
              loc, mlir::util::VarLen32Type::get(rewriter.getContext()), std::get<std::string>(parseResult));
          mlir::Value numericDatum = rt::NumericRuntime::pgx_numeric_from_string(rewriter, loc)({literal})[0];
          rewriter.replaceOp(constantOp, datumToNumericCarrier(rewriter, loc, numericDatum));
          return success();
      }
      if (mlir::isa<mlir::db::PgNumericType>(type)) {
          auto loc = constantOp->getLoc();
          mlir::Value literal = rewriter.create<mlir::util::CreateConstVarLen>(
              loc, mlir::util::VarLen32Type::get(rewriter.getContext()), std::get<std::string>(parseResult));
          mlir::Value numericDatum = rt::NumericRuntime::pgx_numeric_from_string(rewriter, loc)({literal})[0];
          rewriter.replaceOp(constantOp, datumToNumericCarrier(rewriter, loc, numericDatum));
          return success();
      }
      if (auto intType = stdType.dyn_cast_or_null<IntegerType>()) {
          if (type.isa<mlir::db::TimestampType>() || type.isa<mlir::db::DateType>()) {
              int64_t parsedValue = std::get<int64_t>(parseResult);
              int64_t originalValue = parsedValue;

              // Convert date/timestamp to nanoseconds (matching AtLowering behavior)
              if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
                  if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
                      parsedValue *= 86400000000000LL; // Convert days to nanoseconds
                      PGX_LOG(DB_LOWER, DEBUG, "[ConstantLowering] Date constant: days=%lld → nanoseconds=%lld",
                              originalValue, parsedValue);
                  }
              } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
                  // Legacy LingoDB timestamps use nanosecond carriers. PostgreSQL
                  // !db.pg_timestamp constants bypass this branch and stay in
                  // PostgreSQL microseconds.
                  parsedValue *= 1000LL; // Convert microseconds to nanoseconds
                  PGX_LOG(DB_LOWER, DEBUG, "[ConstantLowering] Timestamp constant: microseconds=%lld → nanoseconds=%lld",
                          originalValue, parsedValue);
              }

              rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType,
                                                             rewriter.getIntegerAttr(stdType, parsedValue));
              return success();
          } else if (type.isa<mlir::db::IntervalType>()) {
              // Intervals come in as microseconds, convert to nanoseconds to match date representation
              int64_t microseconds = std::get<int64_t>(parseResult);
              int64_t nanoseconds = microseconds * 1000LL; // Convert microseconds to nanoseconds
              PGX_LOG(DB_LOWER, DEBUG, "[ConstantLowering] Interval constant: microseconds=%lld → nanoseconds=%lld",
                      microseconds, nanoseconds);
              rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType,
                                                             rewriter.getIntegerAttr(stdType, nanoseconds));
              return success();
          } else {
              rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                  constantOp, stdType, rewriter.getIntegerAttr(stdType, std::get<int64_t>(parseResult)));
              return success();
          }
      } else if (auto floatType = stdType.dyn_cast_or_null<FloatType>()) {
         rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, stdType, rewriter.getFloatAttr(stdType, std::get<double>(parseResult)));
         return success();
      } else if (type.isa<mlir::db::StringType>()
                 || mlir::isa<mlir::db::PgTextType, mlir::db::PgVarcharType, mlir::db::PgBpcharType>(type))
      {
          std::string str = std::get<std::string>(parseResult);

          rewriter.replaceOpWithNewOp<mlir::util::CreateConstVarLen>(
              constantOp, mlir::util::VarLen32Type::get(rewriter.getContext()), rewriter.getStringAttr(str));
          return success();
      } else {
          return failure();
      }
      return failure();
   }
};
class CmpOpLowering : public OpConversionPattern<mlir::db::CmpOp> {
   public:
   using OpConversionPattern<mlir::db::CmpOp>::OpConversionPattern;
   arith::CmpIPredicate translateIPredicate(db::DBCmpPredicate pred) const {
      switch (pred) {
         case db::DBCmpPredicate::eq:
            return arith::CmpIPredicate::eq;
         case db::DBCmpPredicate::neq:
            return arith::CmpIPredicate::ne;
         case db::DBCmpPredicate::lt:
            return arith::CmpIPredicate::slt;
         case db::DBCmpPredicate::gt:
            return arith::CmpIPredicate::sgt;
         case db::DBCmpPredicate::lte:
            return arith::CmpIPredicate::sle;
         case db::DBCmpPredicate::gte:
            return arith::CmpIPredicate::sge;
      }
      assert(false && "unexpected case");
      return arith::CmpIPredicate::eq;
   }
   arith::CmpFPredicate translateFPredicate(db::DBCmpPredicate pred) const {
      switch (pred) {
         case db::DBCmpPredicate::eq:
            return arith::CmpFPredicate::OEQ;
         case db::DBCmpPredicate::neq:
            return arith::CmpFPredicate::ONE;
         case db::DBCmpPredicate::lt:
            return arith::CmpFPredicate::OLT;
         case db::DBCmpPredicate::gt:
            return arith::CmpFPredicate::OGT;
         case db::DBCmpPredicate::lte:
            return arith::CmpFPredicate::OLE;
         case db::DBCmpPredicate::gte:
            return arith::CmpFPredicate::OGE;
      }
      assert(false && "unexpected case");
      return arith::CmpFPredicate::OEQ;
   }
   LogicalResult matchAndRewrite(mlir::db::CmpOp cmpOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
       const bool leftIsPgDate = isPgDateValueType(cmpOp.getLeft().getType());
       const bool rightIsPgDate = isPgDateValueType(cmpOp.getRight().getType());
       const bool leftIsPgTimestamp = isPgTimestampValueType(cmpOp.getLeft().getType());
       const bool rightIsPgTimestamp = isPgTimestampValueType(cmpOp.getRight().getType());
       if ((leftIsPgDate && rightIsPgTimestamp) || (leftIsPgTimestamp && rightIsPgDate)) {
           auto loc = cmpOp->getLoc();
           auto leftOperand = unwrapNullableOperand(rewriter, loc, adaptor.getLeft());
           auto rightOperand = unwrapNullableOperand(rewriter, loc, adaptor.getRight());
           mlir::Value leftPayload = leftIsPgDate ? pgDateDaysToTimestampMicros(rewriter, loc, leftOperand.payload)
                                                  : leftOperand.payload;
           mlir::Value rightPayload = rightIsPgDate ? pgDateDaysToTimestampMicros(rewriter, loc, rightOperand.payload)
                                                    : rightOperand.payload;
           mlir::Value result = rewriter.create<arith::CmpIOp>(loc, translateIPredicate(cmpOp.getPredicate()),
                                                               leftPayload, rightPayload);
           if (mlir::Value isNull = combineNullFlags(rewriter, loc, leftOperand.isNull, rightOperand.isNull)) {
               mlir::Type convertedResultType = typeConverter->convertType(cmpOp.getType());
               if (mlir::isa<mlir::TupleType>(convertedResultType)) {
                   rewriter.replaceOpWithNewOp<mlir::util::PackOp>(cmpOp, convertedResultType,
                                                                   mlir::ValueRange{isNull, result});
                   return success();
               }
               mlir::Value falseValue = rewriter.create<arith::ConstantOp>(
                   loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
               result = rewriter.create<arith::SelectOp>(loc, isNull, falseValue, result);
           }
           rewriter.replaceOp(cmpOp, result);
           return success();
       }
       // PGX-LOWER: decimal operands are PG Numeric datums (i64), so they must be
       // compared via numeric_cmp, NOT a raw integer compare (which would compare
       // pointers). Detect via the ORIGINAL db operand type, before the generic
       // int/float path — post-conversion a decimal-as-i64 is indistinguishable
       // from a real i64.
       if (isNumericCarrierType(cmpOp.getLeft().getType()) && isNumericCarrierType(cmpOp.getRight().getType())) {
           // Operands are Numeric datum carriers; extract and numeric_cmp.
           auto loc = cmpOp->getLoc();
           NumericOperand leftOperand = unwrapNumericOperand(rewriter, loc, adaptor.getLeft());
           NumericOperand rightOperand = unwrapNumericOperand(rewriter, loc, adaptor.getRight());
           mlir::Value numericZero = numericIntCarrier(rewriter, loc, 0);
           mlir::Value leftPayload = safeNumericPayload(rewriter, loc, leftOperand, numericZero);
           mlir::Value rightPayload = safeNumericPayload(rewriter, loc, rightOperand, numericZero);
           mlir::Value l = numericCarrierToDatum(rewriter, loc, leftPayload);
           mlir::Value r = numericCarrierToDatum(rewriter, loc, rightPayload);
           mlir::Value cmp = rt::NumericRuntime::pgx_numeric_cmp(rewriter, loc)({l, r})[0];
           mlir::Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(),
                                                                 rewriter.getI32IntegerAttr(0));
           mlir::Value result = rewriter.create<arith::CmpIOp>(loc, translateIPredicate(cmpOp.getPredicate()), cmp, zero);
           if (mlir::Value isNull = combineNullFlags(rewriter, loc, leftOperand.isNull, rightOperand.isNull)) {
               mlir::Type convertedResultType = typeConverter->convertType(cmpOp.getType());
               if (mlir::isa<mlir::TupleType>(convertedResultType)) {
                   rewriter.replaceOpWithNewOp<mlir::util::PackOp>(cmpOp, convertedResultType,
                                                                   mlir::ValueRange{isNull, result});
                   return success();
               }
               mlir::Value falseValue = rewriter.create<arith::ConstantOp>(
                   loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
               result = rewriter.create<arith::SelectOp>(loc, isNull, falseValue, result);
           }
           rewriter.replaceOp(cmpOp, result);
           return success();
       }
       auto leftOperand = unwrapNumericOperand(rewriter, cmpOp->getLoc(), adaptor.getLeft());
       auto rightOperand = unwrapNumericOperand(rewriter, cmpOp->getLoc(), adaptor.getRight());
       if ((leftOperand.isNull || rightOperand.isNull) && leftOperand.payload.getType() == rightOperand.payload.getType()
           && leftOperand.payload.getType().isIntOrIndexOrFloat())
       {
           auto loc = cmpOp->getLoc();
           mlir::Value result;
           if (leftOperand.payload.getType().isIntOrIndex()) {
               result = rewriter.create<arith::CmpIOp>(loc, translateIPredicate(cmpOp.getPredicate()),
                                                       leftOperand.payload, rightOperand.payload);
           } else {
               result = rewriter.create<arith::CmpFOp>(loc, translateFPredicate(cmpOp.getPredicate()),
                                                       leftOperand.payload, rightOperand.payload);
           }
           mlir::Value isNull = combineNullFlags(rewriter, loc, leftOperand.isNull, rightOperand.isNull);
           mlir::Type convertedResultType = typeConverter->convertType(cmpOp.getType());
           if (mlir::isa<mlir::TupleType>(convertedResultType)) {
               rewriter.replaceOpWithNewOp<mlir::util::PackOp>(cmpOp, convertedResultType,
                                                               mlir::ValueRange{isNull, result});
               return success();
           }
           mlir::Value falseValue = rewriter.create<arith::ConstantOp>(
               loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
           rewriter.replaceOpWithNewOp<arith::SelectOp>(cmpOp, isNull, falseValue, result);
           return success();
       }
      if (!adaptor.getLeft().getType().isIntOrIndexOrFloat()) {
         return failure();
      }
      if (adaptor.getLeft().getType().isIntOrIndex()) {
         rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmpOp, translateIPredicate(cmpOp.getPredicate()), adaptor.getLeft(), adaptor.getRight());
      } else {
         rewriter.replaceOpWithNewOp<arith::CmpFOp>(cmpOp, translateFPredicate(cmpOp.getPredicate()), adaptor.getLeft(), adaptor.getRight());
      }
      return success();
   }
};
class CastNoneOpLowering : public OpConversionPattern<mlir::db::CastOp> {
   public:
   using OpConversionPattern<mlir::db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::CastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto scalarSourceType = op.getVal().getType();
      auto scalarTargetType = op.getType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      if (scalarSourceType.isa<mlir::db::StringType>() || scalarTargetType.isa<mlir::db::StringType>()) return failure();
      if (!convertedSourceType.isa<NoneType>()) {
         return mlir::failure();
      }
      rewriter.replaceOpWithNewOp<mlir::util::UndefOp>(op, convertedTargetType);
      return mlir::success();
   }
};
class CastOpLowering : public OpConversionPattern<mlir::db::CastOp> {
   public:
   using OpConversionPattern<mlir::db::CastOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::CastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      auto loc = op->getLoc();
      auto scalarSourceType = op.getVal().getType();
      auto scalarTargetType = op.getType();
      auto convertedSourceType = typeConverter->convertType(scalarSourceType);
      auto convertedTargetType = typeConverter->convertType(scalarTargetType);
      mlir::Type originalTargetType = scalarTargetType;
      if (scalarSourceType.isa<mlir::db::StringType>() || scalarTargetType.isa<mlir::db::StringType>()) return failure();
      Value value = adaptor.getVal();
      if (scalarSourceType == scalarTargetType) {
         rewriter.replaceOp(op, value);
         return success();
      }
      if (auto nullableSourceType = scalarSourceType.dyn_cast_or_null<db::NullableType>()) {
          mlir::Type sourcePayloadType = nullableSourceType.getType();
          if (mlir::db::isPgValueType(sourcePayloadType) && mlir::db::isPgValueType(scalarTargetType)
              && mlir::db::getPgTypeOid(sourcePayloadType) == mlir::db::getPgTypeOid(scalarTargetType)
              && mlir::db::getPgNullability(scalarTargetType) == mlir::db::PgNullability::Maybe)
          {
              auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, value);
              mlir::Value outerNull = unpacked.getVals()[0];
              mlir::Value payload = unpacked.getVals()[1];
              mlir::Value innerNull;
              if (mlir::db::getPgNullability(sourcePayloadType) == mlir::db::PgNullability::Maybe) {
                  auto inner = rewriter.create<mlir::util::UnPackOp>(loc, payload);
                  innerNull = inner.getVals()[0];
                  payload = inner.getVals()[1];
              }
              mlir::Value isNull = combineNullFlags(rewriter, loc, outerNull, innerNull);
              rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, convertedTargetType, mlir::ValueRange{isNull, payload});
              return success();
          }
          if (auto nullableTargetType = scalarTargetType.dyn_cast_or_null<db::NullableType>()) {
              mlir::Type targetPayloadType = nullableTargetType.getType();
              if (mlir::db::isPgValueType(sourcePayloadType) && mlir::db::isPgValueType(targetPayloadType)
                  && mlir::db::getPgTypeOid(sourcePayloadType) == mlir::db::getPgTypeOid(targetPayloadType))
              {
                  auto unpacked = rewriter.create<mlir::util::UnPackOp>(loc, value);
                  mlir::Value outerNull = unpacked.getVals()[0];
                  mlir::Value payload = unpacked.getVals()[1];
                  mlir::Value innerNull;
                  if (mlir::db::getPgNullability(sourcePayloadType) == mlir::db::PgNullability::Maybe) {
                      auto inner = rewriter.create<mlir::util::UnPackOp>(loc, payload);
                      innerNull = inner.getVals()[0];
                      payload = inner.getVals()[1];
                  }

                  if (mlir::db::getPgNullability(targetPayloadType) == mlir::db::PgNullability::Maybe) {
                      if (!innerNull) {
                          innerNull = rewriter.create<arith::ConstantOp>(
                              loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
                      }
                      payload = rewriter.create<mlir::util::PackOp>(loc, typeConverter->convertType(targetPayloadType),
                                                                    mlir::ValueRange{innerNull, payload});
                  } else if (innerNull) {
                      outerNull = rewriter.create<arith::OrIOp>(loc, outerNull, innerNull);
                  }

                  rewriter.replaceOpWithNewOp<mlir::util::PackOp>(op, convertedTargetType,
                                                                  mlir::ValueRange{outerNull, payload});
                  return success();
              }
          }
      }

      bool needsNullableWrap = false;
      mlir::Value nullableWrapIsNull;
      if (mlir::db::isPgValueType(scalarSourceType)
          && mlir::db::getPgNullability(scalarSourceType) == mlir::db::PgNullability::Maybe)
      {
          if (!mlir::db::isPgValueType(scalarTargetType)
              || mlir::db::getPgNullability(scalarTargetType) != mlir::db::PgNullability::Maybe)
          {
              return failure();
          }
          auto nullableValue = unwrapNullableOperand(rewriter, loc, value);
          value = nullableValue.payload;
          nullableWrapIsNull = nullableValue.isNull;
          scalarSourceType = mlir::db::withPgNullability(scalarSourceType, mlir::db::PgNullability::Never);
          scalarTargetType = mlir::db::withPgNullability(scalarTargetType, mlir::db::PgNullability::Never);
          convertedSourceType = typeConverter->convertType(scalarSourceType);
          convertedTargetType = typeConverter->convertType(scalarTargetType);
          needsNullableWrap = true;
      }

      // Support null casting
      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (!scalarSourceType.isa<db::NullableType>()) {
          if (auto nullableTargetType = scalarTargetType.dyn_cast_or_null<db::NullableType>()) {
              scalarTargetType = nullableTargetType.getType();
              convertedTargetType = typeConverter->convertType(scalarTargetType);
              needsNullableWrap = true;
          }
      }

      // Lambda to finalize cast: wraps in nullable if needed, replaces op, returns success
      auto finishCast = [&](const Value resultValue) -> LogicalResult {
          if (needsNullableWrap) {
              auto nullableTupleType = typeConverter->convertType(originalTargetType);
              if (!nullableWrapIsNull) {
                  nullableWrapIsNull = rewriter.create<mlir::arith::ConstantOp>(
                      loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              }
              const Value packed = rewriter.create<mlir::util::PackOp>(
                  loc, nullableTupleType, mlir::ValueRange{nullableWrapIsNull, resultValue});
              rewriter.replaceOp(op, packed);
          } else {
              rewriter.replaceOp(op, resultValue);
          }
          return success();
      };
      auto safeNullableNumericPayload = [&](Value payload) -> Value {
          if (!nullableWrapIsNull) {
              return payload;
          }
          return safeNumericPayload(rewriter, loc, NumericOperand{payload, nullableWrapIsNull},
                                    numericIntCarrier(rewriter, loc, 0));
      };
      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      // Handle Date/Interval casting - they both convert to i64, so no actual conversion needed
      bool sourceIsDateTime = scalarSourceType.isa<mlir::db::DateType, mlir::db::IntervalType, mlir::db::TimestampType>();
      bool targetIsDateTime = scalarTargetType.isa<mlir::db::DateType, mlir::db::IntervalType, mlir::db::TimestampType>();
      if (sourceIsDateTime && targetIsDateTime) {
          // Both convert to i64, so no conversion operation needed
          return finishCast(value);
      }
      if (mlir::db::isPgValueType(scalarSourceType) && mlir::db::isPgValueType(scalarTargetType)
          && mlir::db::getPgTypeOid(scalarSourceType) == mlir::db::getPgTypeOid(scalarTargetType))
      {
          return finishCast(value);
      }
      if (isPgStringValueType(scalarSourceType) && isPgStringValueType(scalarTargetType)) {
          return finishCast(value);
      }
      if (mlir::db::isPgValueType(scalarSourceType) && mlir::db::isPgValueType(scalarTargetType)) {
          mlir::Type nonNullableSourceType = mlir::db::withPgNullability(scalarSourceType,
                                                                         mlir::db::PgNullability::Never);
          mlir::Type nonNullableTargetType = mlir::db::withPgNullability(scalarTargetType,
                                                                         mlir::db::PgNullability::Never);
          constexpr int64_t pgMicrosecondsPerDay = 86400000000LL;
          if (mlir::isa<mlir::db::PgTimestampType>(nonNullableSourceType)
              && mlir::isa<mlir::db::PgDateType>(nonNullableTargetType))
          {
              mlir::Value divisor = rewriter.create<arith::ConstantIntOp>(loc, pgMicrosecondsPerDay, 64);
              value = rewriter.create<arith::DivSIOp>(loc, rewriter.getI64Type(), value, divisor);
              value = castPhysicalScalar(rewriter, loc, value, convertedTargetType);
              return finishCast(value);
          }
          if (mlir::isa<mlir::db::PgDateType>(nonNullableSourceType)
              && mlir::isa<mlir::db::PgTimestampType>(nonNullableTargetType))
          {
              if (value.getType().isInteger(32)) {
                  value = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
              }
              mlir::Value multiplier = rewriter.create<arith::ConstantIntOp>(loc, pgMicrosecondsPerDay, 64);
              value = rewriter.create<arith::MulIOp>(loc, value, multiplier);
              return finishCast(value);
          }
      }

      if (auto sourceIntWidth = getSignedIntegerCarrierWidth(scalarSourceType)) {
          if (getFloatCarrierType(scalarTargetType)) {
              value = rewriter.create<arith::SIToFPOp>(loc, convertedTargetType, value);
              return finishCast(value);
          } else if (isNumericCarrierType(scalarTargetType)) {
              // PGX-LOWER: int -> NUMERIC via PG int8_numeric. Widen source to i64
              // for the stub; result is a datum carrier.
              value = safePayloadOr(rewriter, loc, value, nullableWrapIsNull, 0);
              if (value.getType() != rewriter.getI64Type()) {
                  value = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
              }
              Value datum = rt::NumericRuntime::pgx_int_to_numeric(rewriter, loc)({value})[0];
              return finishCast(datumToNumericCarrier(rewriter, loc, datum));
          } else if (auto targetIntWidth = getSignedIntegerCarrierWidth(scalarTargetType)) {
              Value result;
              if (targetIntWidth < sourceIntWidth) {
                  result = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, value);
              } else if (targetIntWidth > sourceIntWidth) {
                  result = rewriter.create<arith::ExtSIOp>(loc, convertedTargetType, value);
              } else {
                  // Same width - no conversion needed
                  result = value;
              }
              return finishCast(result);
          }
      } else if (auto floatType = getFloatCarrierType(scalarSourceType)) {
          if (getSignedIntegerCarrierWidth(scalarTargetType)) {
              value = rewriter.create<arith::FPToSIOp>(loc, convertedTargetType, value);
              return finishCast(value);
          } else if (isNumericCarrierType(scalarTargetType)) {
              // PGX-LOWER: float -> NUMERIC via PG float8_numeric. Widen to f64 for
              // the stub; result is a datum carrier.
              value = safePayloadOr(rewriter, loc, value, nullableWrapIsNull, 0);
              if (value.getType() != rewriter.getF64Type()) {
                  value = rewriter.create<arith::ExtFOp>(loc, rewriter.getF64Type(), value);
              }
              Value datum = rt::NumericRuntime::pgx_float_to_numeric(rewriter, loc)({value})[0];
              return finishCast(datumToNumericCarrier(rewriter, loc, datum));
          } else if (auto targetFloatType = getFloatCarrierType(scalarTargetType)) {
              // PGX-LOWER edit: Lingodb didn't have type cast for float -> float implemented
              const auto sourceWidth = floatType.getWidth();
              const auto targetWidth = targetFloatType.getWidth();
              Value result;
              if (sourceWidth < targetWidth) {
                  result = rewriter.create<arith::ExtFOp>(loc, convertedTargetType, value);
              } else if (sourceWidth > targetWidth) {
                  result = rewriter.create<arith::TruncFOp>(loc, convertedTargetType, value);
              } else {
                  result = value;
              }
              return finishCast(result);
          }
      } else if (isNumericCarrierType(scalarSourceType)) {
          if (isNumericCarrierType(scalarTargetType)) {
              // PGX-LOWER: NUMERIC -> NUMERIC. A PG Numeric datum carries its own
              // scale, so a scale change is not a representation change — pass the
              // datum through. (PG re-derives scale on output; downstream ops use
              // numeric_* which are scale-aware.)
              return finishCast(value);
          } else if (getFloatCarrierType(scalarTargetType)) {
              // PGX-LOWER: NUMERIC -> float via PG numeric_float8 (returns f64),
              // then narrow if the target is f32. Source is Numeric carrier -> datum.
              Value datum = numericCarrierToDatum(rewriter, loc, safeNullableNumericPayload(value));
              Value result = rt::NumericRuntime::pgx_numeric_to_float(rewriter, loc)({datum})[0];
              if (convertedTargetType != rewriter.getF64Type()) {
                  result = rewriter.create<arith::TruncFOp>(loc, convertedTargetType, result);
              }
              return finishCast(result);
          } else if (auto targetIntWidth = getSignedIntegerCarrierWidth(scalarTargetType)) {
              // PGX-LOWER: NUMERIC -> int via PG numeric_int8 (returns i64), then
              // narrow to the target integer width. Source is Numeric carrier -> datum.
              Value datum = numericCarrierToDatum(rewriter, loc, safeNullableNumericPayload(value));
              Value result = rt::NumericRuntime::pgx_numeric_to_int(rewriter, loc)({datum})[0];
              if (targetIntWidth < 64) {
                  result = rewriter.create<arith::TruncIOp>(loc, convertedTargetType, result);
              }
              return finishCast(result);
          }
      } else if (auto timestampSourceType = scalarSourceType.dyn_cast_or_null<db::TimestampType>()) {
          if (auto dateTargetType = scalarTargetType.dyn_cast_or_null<db::DateType>()) {
              // Both timestamp and date are in nanoseconds, just divide by nanoseconds per day
              const uint64_t nanosecondsPerDay = 86400000000000ULL;
              mlir::Value divisor = rewriter.create<arith::ConstantIntOp>(loc, nanosecondsPerDay, 64);
              value = rewriter.create<arith::DivSIOp>(loc, rewriter.getI64Type(), value, divisor);
              return finishCast(value);
          }
      } else if (auto dateSourceType = scalarSourceType.dyn_cast_or_null<db::DateType>()) {
          if (auto timestampTargetType = scalarTargetType.dyn_cast_or_null<db::TimestampType>()) {
              // Both date and timestamp are in nanoseconds, no conversion needed
              return finishCast(value);
          }
      }

      // PGX-LOWER added log: log when we fail to cast, since it's a common problem and a pain to figure out what cast it
      // was
      {
         std::string opDump;
         llvm::raw_string_ostream os(opDump);
         op->print(os);
         os.flush();

         std::string fromType;
         llvm::raw_string_ostream fromOs(fromType);
         op.getVal().getType().print(fromOs);
         fromOs.flush();

         std::string toType;
         llvm::raw_string_ostream toOs(toType);
         originalTargetType.print(toOs);
         toOs.flush();

         PGX_WARNING("Failed to lower db.cast operation - From type: %s, To type: %s",
                     fromType.c_str(), toType.c_str());
         PGX_WARNING("Full operation dump: %s", opDump.c_str());
      }

      return failure();
   }
};
class BetweenLowering : public OpConversionPattern<mlir::db::BetweenOp> {
   public:
   using OpConversionPattern<mlir::db::BetweenOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::BetweenOp betweenOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
       auto isGteLower = rewriter.create<mlir::db::CmpOp>(betweenOp->getLoc(),
                                                          betweenOp.getLowerInclusive() ? mlir::db::DBCmpPredicate::gte
                                                                                        : mlir::db::DBCmpPredicate::gt,
                                                          adaptor.getVal(), adaptor.getLower());
       auto isLteUpper = rewriter.create<mlir::db::CmpOp>(betweenOp->getLoc(),
                                                          betweenOp.getUpperInclusive() ? mlir::db::DBCmpPredicate::lte
                                                                                        : mlir::db::DBCmpPredicate::lt,
                                                          adaptor.getVal(), adaptor.getUpper());
       auto isInRange = rewriter.create<mlir::db::AndOp>(betweenOp->getLoc(), ValueRange({isGteLower, isLteUpper}));
       rewriter.replaceOp(betweenOp, isInRange.getRes());
       return success();
   }
};
class OneOfLowering : public OpConversionPattern<mlir::db::OneOfOp> {
   public:
   using OpConversionPattern<mlir::db::OneOfOp>::OpConversionPattern;
   LogicalResult matchAndRewrite(mlir::db::OneOfOp oneOfOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
      std::vector<Value> compared;
      for (auto ele : adaptor.getVals()) {
          compared.push_back(
              rewriter.create<mlir::db::CmpOp>(oneOfOp->getLoc(), mlir::db::DBCmpPredicate::eq, adaptor.getVal(), ele));
      }
      auto isInRange = rewriter.create<mlir::db::OrOp>(oneOfOp->getLoc(), compared);
      rewriter.replaceOp(oneOfOp, isInRange.getRes());
      return success();
   }
};
class HashLowering : public ConversionPattern {
   Value combineHashes(OpBuilder& builder, Location loc, Value hash1, Value totalHash) const {
      if (!totalHash) {
         return hash1;
      } else {
         return builder.create<mlir::util::HashCombine>(loc, builder.getIndexType(), hash1, totalHash);
      }
   }
   Value hashInteger(OpBuilder& builder, Location loc, Value integer) const {
      Value asIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), integer);
      return builder.create<mlir::util::Hash64>(loc, builder.getIndexType(), asIndex);
   }
   Value hashImpl(OpBuilder& builder, Location loc, Value v, Value totalHash, Type originalType) const {
       if (v.getType().isa<mlir::IntegerType>() && isNumericCarrierType(getBaseType(originalType))) {
           Value datum = numericCarrierToDatum(builder, loc, v);
           Value hash = rt::NumericRuntime::pgx_numeric_hash(builder, loc)({datum})[0];
           Value asIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), hash);
           return combineHashes(builder, loc, asIndex, totalHash);
       }

      if (auto intType = v.getType().dyn_cast_or_null<mlir::IntegerType>()) {
         if (intType.getWidth() == 128) {
             // Generic 128-bit integer hash path. DecimalType is handled above
             // through PostgreSQL numeric_hash on the Datum carrier.
             auto i64Type = IntegerType::get(builder.getContext(), 64);
             auto i128Type = IntegerType::get(builder.getContext(), 128);

             Value low = builder.create<arith::TruncIOp>(loc, i64Type, v);
             Value shift = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(i128Type, 64));
             Value high = builder.create<arith::ShRUIOp>(loc, i128Type, v, shift);
             Value first = hashInteger(builder, loc, high);
             Value second = hashInteger(builder, loc, low);
             Value combined1 = combineHashes(builder, loc, first, totalHash);
             Value combined2 = combineHashes(builder, loc, second, combined1);
             return combined2;
         } else {
            return combineHashes(builder, loc, hashInteger(builder, loc, v), totalHash);
         }

      } else if (auto floatType = v.getType().dyn_cast_or_null<mlir::FloatType>()) {
         assert(false && "can not hash float values");
      } else if (auto varLenType = v.getType().dyn_cast_or_null<mlir::util::VarLen32Type>()) {
          (void)varLenType;
          mlir::Type baseOriginalType = getBaseType(originalType);
          const uint32_t pgHashFunctionOid = pgStringHashFunctionOid(baseOriginalType);
          if (mlir::db::isPgValueType(baseOriginalType) && pgHashFunctionOid != InvalidOid) {
              Value typeOid = builder.create<arith::ConstantIntOp>(loc, mlir::db::getPgTypeOid(baseOriginalType), 32);
              Value functionOid = builder.create<arith::ConstantIntOp>(loc, pgHashFunctionOid, 32);
              Value collationOid = builder.create<arith::ConstantIntOp>(loc, mlir::db::getPgCollation(baseOriginalType),
                                                                        32);
              Value hash = rt::StringRuntime::pgCallHash1(builder, loc)({v, typeOid, functionOid, collationOid})[0];
              Value asIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), hash);
              return combineHashes(builder, loc, asIndex, totalHash);
          }
         auto hash = builder.create<mlir::util::HashVarLen>(loc, builder.getIndexType(), v);
         return combineHashes(builder, loc, hash, totalHash);
      } else if (auto tupleType = v.getType().dyn_cast_or_null<mlir::TupleType>()) {
         if (auto originalTupleType = originalType.dyn_cast_or_null<mlir::TupleType>()) {
            auto unpacked = builder.create<util::UnPackOp>(loc, v);
            size_t i = 0;
            for (auto v : unpacked->getResults()) {
               totalHash = hashImpl(builder, loc, v, totalHash, originalTupleType.getType(i++));
            }
            if (!totalHash) {
               totalHash = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
            }
            return totalHash;
         } else if (originalType.isa<mlir::db::NullableType>()) {
            auto unpacked = builder.create<util::UnPackOp>(loc, v);
            mlir::Value hashedIfNotNull = hashImpl(builder, loc, unpacked.getResult(1), totalHash, getBaseType(originalType));
            if (!totalHash) {
               totalHash = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
            }
            return builder.create<mlir::arith::SelectOp>(loc, unpacked.getResult(0), totalHash, hashedIfNotNull);
         } else if (mlir::db::isPgValueType(originalType)
                    && mlir::db::getPgNullability(originalType) == mlir::db::PgNullability::Maybe)
         {
             auto unpacked = builder.create<util::UnPackOp>(loc, v);
             mlir::Type nonNullableType = mlir::db::withPgNullability(originalType, mlir::db::PgNullability::Never);
             mlir::Value hashedIfNotNull = hashImpl(builder, loc, unpacked.getResult(1), totalHash, nonNullableType);
             if (!totalHash) {
                 totalHash = builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0));
             }
             return builder.create<mlir::arith::SelectOp>(loc, unpacked.getResult(0), totalHash, hashedIfNotNull);
         }
         assert(false && "should not happen");
         return Value();
      }
      assert(false && "should not happen");
      return Value();
   }

   public:
   explicit HashLowering(TypeConverter& typeConverter, MLIRContext* context)
      : ConversionPattern(typeConverter, mlir::db::Hash::getOperationName(), 1, context) {}
   LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
      mlir::db::HashAdaptor hashAdaptor(operands);
      auto hashOp = mlir::cast<mlir::db::Hash>(op);

      rewriter.replaceOp(op, hashImpl(rewriter, op->getLoc(), hashAdaptor.getVal(), Value(), hashOp.getVal().getType()));
      return success();
   }
};
void DBToStdLoweringPass::runOnOperation() {
   auto module = getOperation();
   getContext().getLoadedDialect<mlir::util::UtilDialect>()->getFunctionHelper().setParentModule(module);

   // Define Conversion Target
   ConversionTarget target(getContext());
   target.addLegalOp<ModuleOp>();
   target.addLegalOp<UnrealizedConversionCastOp>();

   target.addLegalDialect<func::FuncDialect>();
   target.addLegalDialect<memref::MemRefDialect>();
   TypeConverter typeConverter;
   auto *ctxt = &getContext();
   typeConverter.addConversion([&](mlir::Type type) {
       if (mlir::db::isPgValueType(type)) {
           mlir::Type payloadType = mlir::db::getPgPhysicalCarrierType(type);
           if (mlir::db::getPgNullability(type) == mlir::db::PgNullability::Maybe) {
               return (mlir::Type)mlir::TupleType::get(ctxt, {mlir::IntegerType::get(ctxt, 1), payloadType});
           }
           return payloadType;
       }
       return type;
   });
   typeConverter.addConversion([&](::mlir::db::DateType t) {
      return mlir::IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](::mlir::db::DecimalType t) {
       // PGX-LOWER: NUMERIC is represented as a Datum-width PostgreSQL Numeric
       // carrier. Arithmetic/compare/casts/constants call PG-native numeric_*
       // runtime functions; nothing converts datum<->scaled-integer.
       return mlir::IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](::mlir::db::CharType t) {
      if (t.getBytes() > 8) return mlir::Type();
      return (Type) mlir::IntegerType::get(ctxt, t.getBytes() * 8);
   });
   typeConverter.addConversion([&](::mlir::db::StringType t) {
      return mlir::util::VarLen32Type::get(ctxt);
   });
   typeConverter.addConversion([&](::mlir::db::TimestampType t) {
      return mlir::IntegerType::get(ctxt, 64);
   });
   typeConverter.addConversion([&](::mlir::db::IntervalType t) {
       if (t.getUnit() == mlir::db::IntervalUnitAttr::daytime) {
           return mlir::IntegerType::get(ctxt, 64);
       } else {
           return mlir::IntegerType::get(ctxt, 32);
       }
   });
   typeConverter.addConversion([&](mlir::db::NullableType type) {
      mlir::Type payloadType = typeConverter.convertType(type.getType());
      if (payloadType.isa<mlir::NoneType>()) {
         payloadType = IntegerType::get(ctxt, 1);
      }
      return (Type) TupleType::get(ctxt, {IntegerType::get(ctxt, 1), payloadType});
   });
   auto opIsWithoutDBTypes = [&](Operation* op) { return !hasDBType(typeConverter, op->getOperandTypes()) && !hasDBType(typeConverter, op->getResultTypes()); };
   target.addDynamicallyLegalDialect<scf::SCFDialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalDialect<dsa::DSADialect>(opIsWithoutDBTypes);
   target.addDynamicallyLegalDialect<arith::ArithDialect>(opIsWithoutDBTypes);

   target.addLegalDialect<cf::ControlFlowDialect>();

   target.addDynamicallyLegalDialect<util::UtilDialect>(opIsWithoutDBTypes);
   target.addLegalOp<mlir::dsa::CondSkipOp>();

   target.addDynamicallyLegalOp<mlir::dsa::CondSkipOp>(opIsWithoutDBTypes);
   target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto isLegal = !hasDBType(typeConverter, op.getFunctionType().getInputs()) &&
         !hasDBType(typeConverter, op.getFunctionType().getResults());
      return isLegal;
   });
   target.addDynamicallyLegalOp<mlir::func::ConstantOp>([&](mlir::func::ConstantOp op) {
      if (auto functionType = op.getType().dyn_cast_or_null<mlir::FunctionType>()) {
         auto isLegal = !hasDBType(typeConverter, functionType.getInputs()) &&
            !hasDBType(typeConverter, functionType.getResults());
         return isLegal;
      } else {
         return true;
      }
   });
   target.addDynamicallyLegalOp<mlir::func::CallOp, mlir::func::CallIndirectOp, mlir::func::ReturnOp>(opIsWithoutDBTypes);

   target.addDynamicallyLegalOp<util::SizeOfOp>(
      [&typeConverter](util::SizeOfOp op) {
         auto isLegal = !hasDBType(typeConverter, op.getType());
         return isLegal;
      });

   typeConverter.addConversion([&](mlir::TupleType tupleType) {
      return convertTuple(tupleType, typeConverter);
   });

   auto convertPhysical = [&](mlir::TupleType tuple) -> mlir::TupleType {
      std::vector<mlir::Type> types;
      for (auto t : tuple.getTypes()) {
         mlir::Type arrowPhysicalType = typeConverter.convertType(t);
         types.push_back(arrowPhysicalType);
      }
      return mlir::TupleType::get(tuple.getContext(), types);
   };
   typeConverter.addConversion([&](mlir::dsa::RecordType r) {
      return mlir::dsa::RecordType::get(r.getContext(), convertPhysical(r.getRowType()));
   });
   typeConverter.addConversion([&](mlir::dsa::RecordBatchType r) {
      return mlir::dsa::RecordBatchType::get(r.getContext(), convertPhysical(r.getRowType()));
   });
   typeConverter.addConversion([&](mlir::dsa::GenericIterableType r) { return mlir::dsa::GenericIterableType::get(r.getContext(), typeConverter.convertType(r.getElementType()), r.getIteratorName()); });
   typeConverter.addConversion([&](mlir::dsa::VectorType r) { return mlir::dsa::VectorType::get(r.getContext(), typeConverter.convertType(r.getElementType())); });
   typeConverter.addConversion([&](mlir::dsa::JoinHashtableType r) { return mlir::dsa::JoinHashtableType::get(r.getContext(), typeConverter.convertType(r.getKeyType()).cast<mlir::TupleType>(), typeConverter.convertType(r.getValType()).cast<mlir::TupleType>()); });
   typeConverter.addConversion([&](mlir::dsa::AggregationHashtableType r) { return mlir::dsa::AggregationHashtableType::get(r.getContext(), typeConverter.convertType(r.getKeyType()).cast<mlir::TupleType>(), typeConverter.convertType(r.getValType()).cast<mlir::TupleType>()); });
   typeConverter.addConversion([&](mlir::dsa::TableBuilderType r) { return mlir::dsa::TableBuilderType::get(r.getContext(), typeConverter.convertType(r.getRowType()).cast<mlir::TupleType>()); });

   RewritePatternSet patterns(&getContext());

   mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
   mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
   mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
   mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);
   mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
   patterns.insert<SimpleTypeConversionPattern<mlir::func::ConstantOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::arith::SelectOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::CondSkipOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::ScanSource>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::Append>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::SetDecimalScaleOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::CreateDS>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::Finalize>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::Lookup>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::FreeOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::YieldOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::NextRow>>(typeConverter, &getContext());
   patterns.insert<AtLowering>(typeConverter, &getContext());
   patterns.insert<AppendTBLowering>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::HashtableInsert>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::SortOp>>(typeConverter, &getContext());
   patterns.insert<SimpleTypeConversionPattern<mlir::dsa::ForOp>>(typeConverter, &getContext());
   patterns.insert<StringCmpOpLowering>(typeConverter, ctxt);
   patterns.insert<StringCastOpLowering>(typeConverter, ctxt);
   patterns.insert<RuntimeCallLowering>(typeConverter, ctxt);
   patterns.insert<CmpOpLowering>(typeConverter, ctxt);
   patterns.insert<BetweenLowering>(typeConverter, ctxt);
   patterns.insert<OneOfLowering>(typeConverter, ctxt);

   patterns.insert<NotOpLowering>(typeConverter, ctxt);
   patterns.insert<DeriveTruthLowering>(typeConverter, ctxt);

   patterns.insert<AndOpLowering>(typeConverter, ctxt);
   patterns.insert<OrOpLowering>(typeConverter, ctxt);

   // Date/interval arithmetic lowering
   patterns.insert<DateIntervalArithmeticLowering<mlir::db::AddOp, arith::AddIOp>>(typeConverter, ctxt);
   patterns.insert<DateIntervalArithmeticLowering<mlir::db::SubOp, arith::SubIOp>>(typeConverter, ctxt);

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::IntegerType, arith::AddIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::IntegerType, arith::SubIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::IntegerType, arith::MulIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::IntegerType, arith::DivSIOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::IntegerType, arith::RemSIOp>>(typeConverter, ctxt);

   patterns.insert<BinOpLowering<mlir::db::AddOp, mlir::FloatType, arith::AddFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::SubOp, mlir::FloatType, arith::SubFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::MulOp, mlir::FloatType, arith::MulFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::DivOp, mlir::FloatType, arith::DivFOp>>(typeConverter, ctxt);
   patterns.insert<BinOpLowering<mlir::db::ModOp, mlir::FloatType, arith::RemFOp>>(typeConverter, ctxt);

   patterns.insert<DecimalBinOpLowering<mlir::db::AddOp, arith::AddIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalBinOpLowering<mlir::db::SubOp, arith::SubIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalBinOpLowering<mlir::db::MulOp, arith::MulIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalOpScaledLowering<mlir::db::DivOp, arith::DivSIOp>>(typeConverter, ctxt);
   patterns.insert<DecimalOpScaledLowering<mlir::db::ModOp, arith::RemSIOp>>(typeConverter, ctxt);

   patterns.insert<NullOpLowering>(typeConverter, ctxt);
   patterns.insert<IsNullOpLowering>(typeConverter, ctxt);
   patterns.insert<AsNullableOpLowering>(typeConverter, ctxt);
   patterns.insert<NullableGetValOpLowering>(typeConverter, ctxt);

   patterns.insert<ConstantLowering>(typeConverter, ctxt);
   patterns.insert<CastOpLowering>(typeConverter, ctxt);
   patterns.insert<CastNoneOpLowering>(typeConverter, ctxt);

   patterns.insert<HashLowering>(typeConverter, ctxt);

   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

std::unique_ptr<mlir::Pass>
mlir::db::createLowerToStdPass() {
   return std::make_unique<DBToStdLoweringPass>();
}
void mlir::db::createLowerDBPipeline(mlir::OpPassManager& pm) {
   pm.addPass(mlir::db::createEliminateNullsPass());
   pm.addPass(mlir::db::createOptimizeRuntimeFunctionsPass());
   pm.addPass(mlir::db::createInjectDecimalScalePass());
   pm.addPass(mlir::db::createLowerToStdPass());
}
void mlir::db::registerDBConversionPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::db::createOptimizeRuntimeFunctionsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::db::createEliminateNullsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::db::createInjectDecimalScalePass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::db::createLowerToStdPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "lower-db",
      "",
      createLowerDBPipeline);
}
