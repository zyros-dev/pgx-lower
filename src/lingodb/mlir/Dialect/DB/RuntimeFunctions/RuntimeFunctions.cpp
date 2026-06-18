#include "mlir/Dialect/Arith/IR/Arith.h"
#include "lingodb/mlir/Dialect/util/FunctionHelper.h"
#include "lingodb/mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "lingodb/mlir/Dialect/DB/Passes.h"
#include "pgx-lower/utility/logging.h"
#include "runtime-defs/DateRuntime.h"
#include "runtime-defs/DumpRuntime.h"
#include "runtime-defs/StringRuntime.h"
#include "runtime-defs/NumericRuntime.h"
#include "runtime-defs/PostgreSQLRuntime.h"
#include "runtime-defs/PrintRuntime.h"

extern "C" {
#include "utils/fmgroids.h"
}

mlir::db::RuntimeFunction* mlir::db::RuntimeFunctionRegistry::lookup(std::string name) {
    return registeredFunctions[name].get();
}
static bool isPgTextBridgeType(::mlir::Type type) {
    type = getBaseType(type);
    return mlir::isa<mlir::db::PgTextType, mlir::db::PgVarcharType>(type);
}
static ::mlir::Value pgLikeImpl(::mlir::OpBuilder& rewriter, ::mlir::ValueRange loweredArguments,
                                ::mlir::TypeRange originalArgumentTypes, ::mlir::Type resType,
                                ::mlir::TypeConverter* typeConverter, ::mlir::Location loc) {
    using namespace mlir;
    if (loweredArguments.size() != 2 || originalArgumentTypes.size() != 2) {
        return Value();
    }

    if (!isPgTextBridgeType(originalArgumentTypes[0]) || !isPgTextBridgeType(originalArgumentTypes[1])) {
        return rt::StringRuntime::like(rewriter, loc)(loweredArguments)[0];
    }

    auto leftType = getBaseType(originalArgumentTypes[0]);
    auto rightType = getBaseType(originalArgumentTypes[1]);
    Value leftTypeOid = rewriter.create<arith::ConstantIntOp>(loc, mlir::db::getPgTypeOid(leftType), 32);
    Value rightTypeOid = rewriter.create<arith::ConstantIntOp>(loc, mlir::db::getPgTypeOid(rightType), 32);
    Value functionOid = rewriter.create<arith::ConstantIntOp>(loc, F_TEXTLIKE, 32);
    Value collationOid = rewriter.create<arith::ConstantIntOp>(loc, mlir::db::getPgCollation(leftType), 32);
    return rt::StringRuntime::pgCallBool2(rewriter, loc)(
        {loweredArguments[0], leftTypeOid, loweredArguments[1], rightTypeOid, functionOid, collationOid})[0];
}
static ::mlir::Value dateAddImpl(::mlir::OpBuilder& rewriter, ::mlir::ValueRange loweredArguments, ::mlir::TypeRange originalArgumentTypes, ::mlir::Type resType, ::mlir::TypeConverter* typeConverter,::mlir::Location loc) {
   using namespace mlir;
   if (llvm::cast<mlir::db::IntervalType>(originalArgumentTypes[1]).getUnit() == mlir::db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::AddIOp>(loc, loweredArguments);
   } else {
      return rt::DateRuntime::addMonths(rewriter, loc)(loweredArguments)[0];
   }
}
static ::mlir::Value absIntImpl(::mlir::OpBuilder& rewriter, ::mlir::ValueRange loweredArguments, ::mlir::TypeRange originalArgumentTypes, ::mlir::Type resType, ::mlir::TypeConverter* typeConverter,::mlir::Location loc) {
   using namespace mlir;
   ::mlir::Value val = loweredArguments[0];
   ::mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, resType, rewriter.getIntegerAttr(resType, 0));
   ::mlir::Value negated = rewriter.create<mlir::arith::SubIOp>(loc, zero, val);
   ::mlir::Value ltZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, val, zero);
   return rewriter.create<mlir::arith::SelectOp>(loc, ltZero, negated, val);
}

static ::mlir::Value absDecimalImpl(::mlir::OpBuilder& rewriter, ::mlir::ValueRange loweredArguments, ::mlir::TypeRange originalArgumentTypes, ::mlir::Type resType, ::mlir::TypeConverter* typeConverter,::mlir::Location loc) {
   using namespace mlir;
   ::mlir::Value val = loweredArguments[0];
   auto valType = val.getType();
   ::mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, valType, rewriter.getIntegerAttr(valType, 0));

   ::mlir::Value ltZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, val, zero);
   ::mlir::Value negated = rewriter.create<mlir::arith::SubIOp>(loc, zero, val);
   return rewriter.create<mlir::arith::SelectOp>(loc, ltZero, negated, val);
}
static ::mlir::Value dateSubImpl(::mlir::OpBuilder& rewriter, ::mlir::ValueRange loweredArguments, ::mlir::TypeRange originalArgumentTypes, ::mlir::Type resType, ::mlir::TypeConverter* typeConverter,::mlir::Location loc) {
   using namespace mlir;
   if (llvm::cast<mlir::db::IntervalType>(originalArgumentTypes[1]).getUnit() == mlir::db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::SubIOp>(loc, loweredArguments);
   } else {
      return rt::DateRuntime::subtractMonths(rewriter, loc)(loweredArguments)[0];
   }
}
static ::mlir::Value matchPart(::mlir::OpBuilder& builder, ::mlir::Location loc, ::mlir::Value lastMatchEnd, std::string pattern, ::mlir::Value str, ::mlir::Value end) {
   if (pattern.empty()) {
      if (!lastMatchEnd) {
         lastMatchEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      }
      return lastMatchEnd;
   }
   ::mlir::Value needleValue = builder.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(builder.getContext()), pattern);
   if (lastMatchEnd) {
      ::mlir::Value matchEnd = rt::StringRuntime::findMatch(builder, loc)(::mlir::ValueRange{str, needleValue, lastMatchEnd, end})[0];
      return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), matchEnd);
   } else {
      ::mlir::Value startsWithPattern = rt::StringRuntime::startsWith(builder, loc)(::mlir::ValueRange{str, needleValue})[0];
      ::mlir::Value patternLen = builder.create<mlir::arith::ConstantIndexOp>(loc, pattern.size());
      ::mlir::Value invalidPos = builder.create<mlir::arith::ConstantIndexOp>(loc, 0x8000000000000000);

      ::mlir::Value matchEnd = builder.create<mlir::arith::SelectOp>(loc, startsWithPattern, patternLen, invalidPos);

      return matchEnd;
   }
}
static ::mlir::Value constLikeImpl(::mlir::OpBuilder& rewriter, ::mlir::ValueRange loweredArguments, ::mlir::TypeRange originalArgumentTypes, ::mlir::Type resType, ::mlir::TypeConverter* typeConverter,::mlir::Location loc) {
   using namespace mlir;
   ::mlir::Value str = loweredArguments[0];
   ::mlir::Value patternValue = loweredArguments[1];
   if (auto constStrOp = mlir::dyn_cast_or_null<mlir::util::CreateConstVarLen>(patternValue.getDefiningOp())) {
      auto pattern = constStrOp.getStr().str();
      size_t pos = 0;
      std::string currentSubPattern;
      ::mlir::Value lastMatchEnd;
      ::mlir::Value end = rewriter.create<util::VarLenGetLen>(loc, rewriter.getIndexType(), str);
      bool flexible=false;
      while (pos < pattern.size()) {
         if (pattern[pos] == '\\') {
            currentSubPattern += pattern[pos + 1];
            pos += 2;
         } else if (pattern[pos] == '_') {
             lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end);
             ::mlir::Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
             lastMatchEnd = rewriter.create<arith::AddIOp>(loc, lastMatchEnd, one);
             currentSubPattern = "";
             pos += 1;
         } else if (pattern[pos] == '%') {
             flexible = true;
             lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end);
             currentSubPattern = "";
             pos += 1;
         } else {
             currentSubPattern += pattern[pos];
             pos += 1;
         }
      }
      if (!currentSubPattern.empty()) {
         ::mlir::Value needleValue = rewriter.create<mlir::util::CreateConstVarLen>(loc, mlir::util::VarLen32Type::get(rewriter.getContext()), currentSubPattern);
         ::mlir::Value endsWith = rt::StringRuntime::endsWith(rewriter, loc)({str, needleValue})[0];
         if (lastMatchEnd) {
            ::mlir::Value patternLength = rewriter.create<mlir::arith::ConstantIndexOp>(loc, currentSubPattern.size());
            lastMatchEnd = rewriter.create<mlir::arith::AddIOp>(loc, lastMatchEnd, patternLength);
            ::mlir::Value previousMatchesEnd = rewriter.create<mlir::arith::CmpIOp>(loc, flexible?arith::CmpIPredicate::ule:arith::CmpIPredicate::eq, lastMatchEnd, end);
            return rewriter.create<mlir::arith::AndIOp>(loc, previousMatchesEnd, endsWith);
         } else {
            return endsWith;
         }
         lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end);
      }

      return rewriter.create<mlir::arith::CmpIOp>(loc, flexible?arith::CmpIPredicate::ule:arith::CmpIPredicate::eq, lastMatchEnd, end);
   }

   return Value();
}
static ::mlir::Value dumpValuesImpl(::mlir::OpBuilder& rewriter, ::mlir::ValueRange loweredArguments, ::mlir::TypeRange originalArgumentTypes, ::mlir::Type resType, ::mlir::TypeConverter* typeConverter,::mlir::Location loc) {
   using namespace mlir;
   auto i64Type = IntegerType::get(rewriter.getContext(), 64);
   auto nullableType = originalArgumentTypes[0].dyn_cast_or_null<mlir::db::NullableType>();
   auto baseType = getBaseType(originalArgumentTypes[0]);

   auto f64Type = rewriter.getF64Type();
   Value isNull;
   Value val;
   if (nullableType) {
      auto unPackOp = rewriter.create<mlir::util::UnPackOp>(loc, loweredArguments[0]);
      isNull = unPackOp.getVals()[0];
      val = unPackOp.getVals()[1];
   } else {
      isNull = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      val = loweredArguments[0];
   }
   if (baseType.isa<mlir::IndexType>()) {
      rt::DumpRuntime::dumpIndex(rewriter, loc)(loweredArguments[0]);
   } else if (isIntegerType(baseType, 1)) {
      rt::DumpRuntime::dumpBool(rewriter, loc)({isNull, val});
   } else if (auto intWidth = getIntegerWidth(baseType, false)) {
      if (intWidth < 64) {
         val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
      }
      rt::DumpRuntime::dumpInt(rewriter, loc)({isNull, val});
   } else if (auto uIntWidth = getIntegerWidth(baseType, true)) {
      if (uIntWidth < 64) {
         val = rewriter.create<arith::ExtUIOp>(loc, i64Type, val);
      }
      rt::DumpRuntime::dumpUInt(rewriter, loc)({isNull, val});
   } else if (baseType.isa<mlir::db::DecimalType>()) {
       Value datum = val;
       if (datum.getType() != i64Type) {
           datum = rewriter.create<arith::TruncIOp>(loc, i64Type, datum);
       }
       rt::DumpRuntime::dumpNumeric(rewriter, loc)({isNull, datum});
   } else if (auto dateType = baseType.dyn_cast_or_null<mlir::db::DateType>()) {
       rt::DumpRuntime::dumpDate(rewriter, loc)({isNull, val});
   } else if (auto timestampType = baseType.dyn_cast_or_null<mlir::db::TimestampType>()) {
       switch (timestampType.getUnit()) {
       case mlir::db::TimeUnitAttr::second: rt::DumpRuntime::dumpTimestampSecond(rewriter, loc)({isNull, val}); break;
       case mlir::db::TimeUnitAttr::millisecond:
           rt::DumpRuntime::dumpTimestampMilliSecond(rewriter, loc)({isNull, val});
           break;
       case mlir::db::TimeUnitAttr::microsecond:
           rt::DumpRuntime::dumpTimestampMicroSecond(rewriter, loc)({isNull, val});
           break;
       case mlir::db::TimeUnitAttr::nanosecond:
           rt::DumpRuntime::dumpTimestampNanoSecond(rewriter, loc)({isNull, val});
           break;
       }
   } else if (auto intervalType = baseType.dyn_cast_or_null<mlir::db::IntervalType>()) {
       if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
           rt::DumpRuntime::dumpIntervalMonths(rewriter, loc)({isNull, val});
       } else {
           rt::DumpRuntime::dumpIntervalDaytime(rewriter, loc)({isNull, val});
       }
   } else if (auto floatType = baseType.dyn_cast_or_null<::mlir::FloatType>()) {
       if (floatType.getWidth() < 64) {
           val = rewriter.create<arith::ExtFOp>(loc, f64Type, val);
       }
       rt::DumpRuntime::dumpFloat(rewriter, loc)({isNull, val});
   } else if (baseType.isa<mlir::db::StringType>()) {
       rt::DumpRuntime::dumpString(rewriter, loc)({isNull, val});
   } else if (auto charType = baseType.dyn_cast_or_null<mlir::db::CharType>()) {
       Value numBytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(charType.getBytes()));
       if (charType.getBytes() < 8) {
           val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
       }
       rt::DumpRuntime::dumpChar(rewriter, loc)({isNull, val, numBytes});
   }
   return ::mlir::Value();
}
std::shared_ptr<mlir::db::RuntimeFunctionRegistry> mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(::mlir::MLIRContext* context) {
   auto builtinRegistry = std::make_shared<RuntimeFunctionRegistry>(context);
   builtinRegistry->add("DumpValue").handlesNulls().matchesTypes({RuntimeFunction::anyType}, RuntimeFunction::noReturnType).implementedAs(dumpValuesImpl);
   auto resTypeIsI64 = [](::mlir::Type t, ::mlir::TypeRange) { return t.isInteger(64); };
   auto resTypeIsI32 = [](::mlir::Type t, ::mlir::TypeRange) { return t.isInteger(32); };
   auto resTypeIsBool = [](::mlir::Type t, ::mlir::TypeRange) {
       return t.isInteger(1) || mlir::isa<mlir::db::PgBoolType>(t);
   };
   auto resTypeIsString = [](::mlir::Type t, ::mlir::TypeRange) { return t.isa<mlir::db::StringType>(); };
   auto resTypeIsStringLike = [](::mlir::Type t, ::mlir::TypeRange) { return RuntimeFunction::stringLike(t); };
   auto i32Like = [](::mlir::Type t) { return t.isInteger(32); };
   auto pgBridgeI32Like = [](::mlir::Type t) {
       t = getBaseType(t);
       return t.isInteger(32) || mlir::isa<mlir::db::PgInt4Type>(t);
   };
   auto rowLike = [](::mlir::Type t) { return mlir::isa<mlir::db::PgRowType>(t); };
   auto resTypeIsRow = [](::mlir::Type t, ::mlir::TypeRange) { return mlir::isa<mlir::db::PgRowType>(t); };
   builtinRegistry->add("PgStringBool2")
       .implementedAs(rt::StringRuntime::pgCallBool2)
       .matchesTypes({RuntimeFunction::stringLike, i32Like, RuntimeFunction::stringLike, i32Like, i32Like, i32Like},
                     resTypeIsBool)
       .needsWrapping();
   builtinRegistry->add("PgStringCall1")
       .implementedAs(rt::StringRuntime::pgCallString1)
       .matchesTypes({RuntimeFunction::stringLike, i32Like, i32Like, i32Like}, resTypeIsStringLike)
       .needsWrapping();
   builtinRegistry->add("PgStringCall2")
       .implementedAs(rt::StringRuntime::pgCallString2)
       .matchesTypes({RuntimeFunction::stringLike, i32Like, pgBridgeI32Like, i32Like, i32Like}, resTypeIsStringLike)
       .needsWrapping();
   builtinRegistry->add("PgStringCall3")
       .implementedAs(rt::StringRuntime::pgCallString3)
       .matchesTypes({RuntimeFunction::stringLike, i32Like, pgBridgeI32Like, pgBridgeI32Like, i32Like, i32Like},
                     resTypeIsStringLike)
       .needsWrapping();
   builtinRegistry->add("Substring")
       .implementedAs(rt::StringRuntime::substr)
       .matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::intLike, RuntimeFunction::intLike},
                     RuntimeFunction::matchesArgument());
   builtinRegistry->add("Like")
       .implementedAs(pgLikeImpl)
       .matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool)
       .needsWrapping();
   builtinRegistry->add("ConstLike")
       .matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool)
       .implementedAs(constLikeImpl)
       .needsWrapping();

   builtinRegistry->add("Concat").implementedAs(rt::StringRuntime::concat).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("Upper").implementedAs(rt::StringRuntime::upper).matchesTypes({RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("Lower").implementedAs(rt::StringRuntime::lower).matchesTypes({RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("ToString").implementedAs(rt::StringRuntime::fromInt).matchesTypes({RuntimeFunction::intLike}, resTypeIsString);

   builtinRegistry->add("ExtractFromDate").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(rt::DateRuntime::ExtractFromDate);
   builtinRegistry->add("ExtractYearFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(rt::DateRuntime::extractYear);
   builtinRegistry->add("ExtractMonthFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(rt::DateRuntime::extractMonth);
   builtinRegistry->add("ExtractDayFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(rt::DateRuntime::extractDay);
   builtinRegistry->add("DateAdd").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateAddImpl);
   builtinRegistry->add("AbsInt").handlesInvalid().matchesTypes({RuntimeFunction::intLike}, RuntimeFunction::matchesArgument()).implementedAs(absIntImpl);
   builtinRegistry->add("AbsDecimal").handlesInvalid().matchesTypes({RuntimeFunction::decimalLike}, RuntimeFunction::matchesArgument()).implementedAs(absDecimalImpl);
   builtinRegistry->add("DateSubtract").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateSubImpl);

   builtinRegistry->add("PgRowScanStart")
       .handlesInvalid()
       .matchesTypes({i32Like}, resTypeIsRow)
       .implementedAs(rt::PgRowRuntime::scanStart);
   builtinRegistry->add("PgRowScanNext")
       .handlesInvalid()
       .matchesTypes({rowLike}, resTypeIsBool)
       .implementedAs(rt::PgRowRuntime::scanNext);
   builtinRegistry->add("PgRowScanEnd")
       .handlesInvalid()
       .matchesTypes({rowLike}, RuntimeFunction::noReturnType)
       .implementedAs(rt::PgRowRuntime::scanEnd);

   // PG-native NUMERIC: decimal arithmetic/compare go through PostgreSQL's own
   // numeric_* functions (full precision, NaN/Inf, any scale) instead of i128.
   builtinRegistry->add("NumericAdd")
       .matchesTypes({RuntimeFunction::decimalLike, RuntimeFunction::decimalLike}, RuntimeFunction::matchesArgument())
       .implementedAs(rt::NumericRuntime::pgx_numeric_add);
   builtinRegistry->add("NumericSub")
       .matchesTypes({RuntimeFunction::decimalLike, RuntimeFunction::decimalLike}, RuntimeFunction::matchesArgument())
       .implementedAs(rt::NumericRuntime::pgx_numeric_sub);
   builtinRegistry->add("NumericMul")
       .matchesTypes({RuntimeFunction::decimalLike, RuntimeFunction::decimalLike}, RuntimeFunction::matchesArgument())
       .implementedAs(rt::NumericRuntime::pgx_numeric_mul);
   builtinRegistry->add("NumericCmp")
       .matchesTypes({RuntimeFunction::decimalLike, RuntimeFunction::decimalLike}, resTypeIsI32)
       .implementedAs(rt::NumericRuntime::pgx_numeric_cmp);

   // Print functions for runtime debugging
   builtinRegistry->add("Print").implementedAs(rt::PrintRuntime::print).matchesTypes({RuntimeFunction::stringLike}, RuntimeFunction::noReturnType);
   builtinRegistry->add("PrintVal").implementedAs(rt::PrintRuntime::printVal).matchesTypes({RuntimeFunction::anyType, RuntimeFunction::intLike}, RuntimeFunction::noReturnType);
   builtinRegistry->add("PrintPtr").implementedAs(rt::PrintRuntime::printPtr).matchesTypes({RuntimeFunction::anyType, RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::noReturnType);

   return builtinRegistry;
}
