#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/Dialect/DB/Passes.h"
#include "runtime/DateRuntime.h"
#include "runtime/DumpRuntime.h"
#include "runtime/StringRuntime.h"

namespace rt = pgx_lower::compiler::runtime;

pgx::mlir::db::RuntimeFunction* pgx::mlir::db::RuntimeFunctionRegistry::lookup(std::string name) {
   return registeredFunctions[name].get();
}
static mlir::Value dateAddImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   if (originalArgumentTypes[1].cast<pgx::mlir::db::IntervalType>().getUnit() == pgx::mlir::db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::AddIOp>(loc, loweredArguments);
   } else {
      // TODO: Fix addMonths wrapper
      // return pgx_lower::compiler::runtime::DateRuntime::addMonths(rewriter, loc)(loweredArguments)[0];
      return loweredArguments[0];
   }
}
static mlir::Value absIntImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   mlir::Value val = loweredArguments[0];
   mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, resType, rewriter.getIntegerAttr(resType, 0));
   mlir::Value negated = rewriter.create<mlir::arith::SubIOp>(loc, zero, val);
   mlir::Value ltZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, val, zero);
   return rewriter.create<mlir::arith::SelectOp>(loc, ltZero, negated, val);
}
static mlir::Value dateSubImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   if (originalArgumentTypes[1].cast<pgx::mlir::db::IntervalType>().getUnit() == pgx::mlir::db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::SubIOp>(loc, loweredArguments);
   } else {
      // TODO: Fix subtractMonths wrapper
      // return pgx_lower::compiler::runtime::DateRuntime::subtractMonths(rewriter, loc)(loweredArguments)[0];
      return loweredArguments[0];
   }
}
static mlir::Value matchPart(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lastMatchEnd, std::string pattern, mlir::Value str, mlir::Value end) {
   if (pattern.empty()) {
      if (!lastMatchEnd) {
         lastMatchEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      }
      return lastMatchEnd;
   }
   mlir::Value needleValue = builder.create<pgx::mlir::util::CreateConstVarLen>(loc, pgx::mlir::util::VarLen32Type::get(builder.getContext()), pattern);
   if (lastMatchEnd) {
      // TODO: Fix findMatch wrapper
      // mlir::Value matchEnd = pgx_lower::compiler::runtime::StringRuntime::findMatch(builder, loc)(mlir::ValueRange{str, needleValue, lastMatchEnd, end})[0];
      mlir::Value matchEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), matchEnd);
   } else {
      // TODO: Fix startsWith wrapper
      // mlir::Value startsWithPattern = pgx_lower::compiler::runtime::StringRuntime::startsWith(builder, loc)(mlir::ValueRange{str, needleValue})[0];
      mlir::Value startsWithPattern = builder.create<mlir::arith::ConstantOp>(loc, builder.getBoolAttr(false));
      mlir::Value patternLen = builder.create<mlir::arith::ConstantIndexOp>(loc, pattern.size());
      mlir::Value invalidPos = builder.create<mlir::arith::ConstantIndexOp>(loc, 0x8000000000000000);

      mlir::Value matchEnd = builder.create<mlir::arith::SelectOp>(loc, startsWithPattern, patternLen, invalidPos);

      return matchEnd;
   }
}
static mlir::Value constLikeImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   mlir::Value str = loweredArguments[0];
   mlir::Value patternValue = loweredArguments[1];
   if (auto constStrOp = mlir::dyn_cast_or_null<pgx::mlir::util::CreateConstVarLen>(patternValue.getDefiningOp())) {
      auto pattern = constStrOp.getStr().str();
      size_t pos = 0;
      std::string currentSubPattern;
      mlir::Value lastMatchEnd;
      mlir::Value end = rewriter.create<pgx::mlir::util::VarLenGetLen>(loc, rewriter.getIndexType(), str);
      bool flexible=false;
      while (pos < pattern.size()) {
         if (pattern[pos] == '\\') {
            currentSubPattern += pattern[pos + 1];
            pos += 2;
         } else if (pattern[pos] == '.') {
            //match current pattern
            lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end);
            mlir::Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

            lastMatchEnd = rewriter.create<arith::AddIOp>(loc, lastMatchEnd, one);
            currentSubPattern = "";
            //lastMatchEnd+=1
            pos += 1;
         } else if (pattern[pos] == '%') {
            flexible=true;
            lastMatchEnd = matchPart(rewriter, loc, lastMatchEnd, currentSubPattern, str, end);
            currentSubPattern = "";
            pos += 1;
         } else {
            currentSubPattern += pattern[pos];
            pos += 1;
         }
      }
      if (!currentSubPattern.empty()) {
         mlir::Value needleValue = rewriter.create<pgx::mlir::util::CreateConstVarLen>(loc, pgx::mlir::util::VarLen32Type::get(rewriter.getContext()), currentSubPattern);
         // TODO: Fix endsWith wrapper
         // mlir::Value endsWith = pgx_lower::compiler::runtime::StringRuntime::endsWith(rewriter, loc)({str, needleValue})[0];
         mlir::Value endsWith = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
         if (lastMatchEnd) {
            mlir::Value patternLength = rewriter.create<mlir::arith::ConstantIndexOp>(loc, currentSubPattern.size());
            lastMatchEnd = rewriter.create<mlir::arith::AddIOp>(loc, lastMatchEnd, patternLength);
            mlir::Value previousMatchesEnd = rewriter.create<mlir::arith::CmpIOp>(loc, flexible?arith::CmpIPredicate::ule:arith::CmpIPredicate::eq, lastMatchEnd, end);
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
static mlir::Value dumpValuesImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, mlir::TypeConverter* typeConverter,mlir::Location loc) {
   using namespace mlir;
   auto i128Type = IntegerType::get(rewriter.getContext(), 128);
   auto i64Type = IntegerType::get(rewriter.getContext(), 64);
   auto nullableType = originalArgumentTypes[0].dyn_cast_or_null<pgx::mlir::db::NullableType>();
   auto baseType = getBaseType(originalArgumentTypes[0]);

   auto f64Type = rewriter.getF64Type();
   Value isNull;
   Value val;
   if (nullableType) {
      auto unPackOp = rewriter.create<pgx::mlir::util::UnPackOp>(loc, loweredArguments[0]);
      isNull = unPackOp.getVals()[0];
      val = unPackOp.getVals()[1];
   } else {
      isNull = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      val = loweredArguments[0];
   }
   if (baseType.isa<mlir::IndexType>()) {
      // TODO: Fix dumpIndex wrapper
      // rt::DumpRuntime::dumpIndex(rewriter, loc)(loweredArguments[0]);
   } else if (isIntegerType(baseType, 1)) {
      // TODO: Fix dumpBool wrapper - needs proper LLVM function call generation
      // rt::DumpRuntime::dumpBool(rewriter, loc)({isNull, val});
   } else if (auto intWidth = getIntegerWidth(baseType, false)) {
      if (intWidth < 64) {
         val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
      }
      // TODO: Fix dumpInt wrapper
      // rt::DumpRuntime::dumpInt(rewriter, loc)({isNull, val});
   } else if (auto uIntWidth = getIntegerWidth(baseType, true)) {
      if (uIntWidth < 64) {
         val = rewriter.create<arith::ExtUIOp>(loc, i64Type, val);
      }
      // TODO: Fix dumpUInt wrapper
      // rt::DumpRuntime::dumpUInt(rewriter, loc)({isNull, val});
   } else if (auto decType = baseType.dyn_cast_or_null<pgx::mlir::db::DecimalType>()) {
      if (typeConverter->convertType(decType).cast<mlir::IntegerType>().getWidth() < 128) {
         auto converted = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(128), val);
         val = converted;
      }
      Value low = rewriter.create<arith::TruncIOp>(loc, i64Type, val);
      Value shift = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i128Type, 64));
      Value scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(decType.getS()));
      Value high = rewriter.create<arith::ShRUIOp>(loc, i128Type, val, shift);
      high = rewriter.create<arith::TruncIOp>(loc, i64Type, high);
      // TODO: Fix dumpDecimal wrapper
      // rt::DumpRuntime::dumpDecimal(rewriter, loc)({isNull, low, high, scale});
   } else if (auto dateType = baseType.dyn_cast_or_null<pgx::mlir::db::DateType>()) {
      // TODO: Fix dumpDate wrapper
      // rt::DumpRuntime::dumpDate(rewriter, loc)({isNull, val});
   } else if (auto timestampType = baseType.dyn_cast_or_null<pgx::mlir::db::TimestampType>()) {
      switch (timestampType.getUnit()) {
         case pgx::mlir::db::TimeUnitAttr::second: break; // TODO: Fix wrapper
         case pgx::mlir::db::TimeUnitAttr::millisecond: break; // TODO: Fix wrapper
         case pgx::mlir::db::TimeUnitAttr::microsecond: break; // TODO: Fix wrapper
         case pgx::mlir::db::TimeUnitAttr::nanosecond: break; // TODO: Fix wrapper
      }
   } else if (auto intervalType = baseType.dyn_cast_or_null<pgx::mlir::db::IntervalType>()) {
      if (intervalType.getUnit() == pgx::mlir::db::IntervalUnitAttr::months) {
         // TODO: Fix wrapper
         // rt::DumpRuntime::dumpIntervalMonths(rewriter, loc)({isNull, val});
      } else {
         // TODO: Fix wrapper
         // rt::DumpRuntime::dumpIntervalDaytime(rewriter, loc)({isNull, val});
      }
   } else if (auto floatType = baseType.dyn_cast_or_null<mlir::FloatType>()) {
      if (floatType.getWidth() < 64) {
         val = rewriter.create<arith::ExtFOp>(loc, f64Type, val);
      }
      // TODO: Fix wrapper
      // rt::DumpRuntime::dumpFloat(rewriter, loc)({isNull, val});
   } else if (baseType.isa<pgx::mlir::db::StringType>()) {
      // TODO: Fix wrapper
      // rt::DumpRuntime::dumpString(rewriter, loc)({isNull, val});
   } else if (auto charType = baseType.dyn_cast_or_null<pgx::mlir::db::CharType>()) {
      Value numBytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(charType.getBytes()));
      if (charType.getBytes() < 8) {
         val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
      }
      // TODO: Fix wrapper
      // rt::DumpRuntime::dumpChar(rewriter, loc)({isNull, val, numBytes});
   }
   return mlir::Value();
}
std::shared_ptr<pgx::mlir::db::RuntimeFunctionRegistry> pgx::mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(mlir::MLIRContext* context) {
   auto builtinRegistry = std::make_shared<RuntimeFunctionRegistry>(context);
   builtinRegistry->add("DumpValue").handlesNulls().matchesTypes({RuntimeFunction::anyType}, RuntimeFunction::noReturnType).implementedAs(dumpValuesImpl);
   auto resTypeIsI64 = [](::mlir::Type t, ::mlir::TypeRange) { return t.isInteger(64); };
   auto resTypeIsBool = [](::mlir::Type t, ::mlir::TypeRange) { return t.isInteger(1); };
   // TODO: Fix implementedAs wrapper for substr
   // builtinRegistry->add("Substring").implementedAs(pgx_lower::compiler::runtime::StringRuntime::substr).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument(0));
   // TODO: Fix implementedAs wrapper for like
   // builtinRegistry->add("Like").implementedAs(pgx_lower::compiler::runtime::StringRuntime::like).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool);
   builtinRegistry->add("ConstLike").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool).implementedAs(constLikeImpl).needsWrapping();

   builtinRegistry->add("ExtractFromDate").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike}, resTypeIsI64);
   // TODO: Fix implementedAs wrapper for extractYear
   // builtinRegistry->add("ExtractYearFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(pgx_lower::compiler::runtime::DateRuntime::extractYear);
   // TODO: Fix implementedAs wrapper for extractMonth
   // builtinRegistry->add("ExtractMonthFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(pgx_lower::compiler::runtime::DateRuntime::extractMonth);
   // TODO: Fix implementedAs wrapper for extractDay
   // builtinRegistry->add("ExtractDayFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(pgx_lower::compiler::runtime::DateRuntime::extractDay);
   builtinRegistry->add("DateAdd").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument(0)).implementedAs(dateAddImpl);
   builtinRegistry->add("AbsInt").handlesInvalid().matchesTypes({RuntimeFunction::intLike}, RuntimeFunction::matchesArgument(0)).implementedAs(absIntImpl);
   builtinRegistry->add("DateSubtract").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument(0)).implementedAs(dateSubImpl);
   return builtinRegistry;
}