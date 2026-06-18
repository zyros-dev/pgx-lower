#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include <unordered_set>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <limits>
#include <queue>

constexpr auto MAX_NUMERIC_PRECISION = 32;
constexpr auto MAX_NUMERIC_UNCONSTRAINED_SCALE = 6;

using namespace mlir;
bool mlir::db::CmpOp::isEqualityPred() { return getPredicate() == mlir::db::DBCmpPredicate::eq; }
bool mlir::db::CmpOp::isLessPred(bool eq) { return getPredicate() == (eq ? mlir::db::DBCmpPredicate::lte : mlir::db::DBCmpPredicate::lt); }
bool mlir::db::CmpOp::isGreaterPred(bool eq) { return getPredicate() == (eq ? mlir::db::DBCmpPredicate::gte : mlir::db::DBCmpPredicate::gt); }
static bool isPgValue(mlir::Type type) {
    return mlir::db::isPgValueType(type);
}
static bool isPgValue(mlir::Value value) {
    return isPgValue(value.getType());
}
static bool hasPgValue(mlir::ValueRange values) {
    return llvm::any_of(values, [](mlir::Value value) { return isPgValue(value); });
}
static Type wrapNullableType(MLIRContext* context, Type type, ValueRange values) {
   if (llvm::any_of(values, [](Value v) { return v.getType().isa<mlir::db::NullableType>(); })) {
      return mlir::db::NullableType::get(type);
   }
   return type;
}
Type mlir::db::inferLogicalResultType(MLIRContext* context, ValueRange values) {
    if (hasPgValue(values)) {
        return mlir::db::PgBoolType::get(context, mlir::db::combineSqlNullability(values));
    }
    return wrapNullableType(context, IntegerType::get(context, 1), values);
}
static Type inferLegacyLogicalResultType(MLIRContext* context, ValueRange values) {
    return mlir::db::inferLogicalResultType(context, values);
}
mlir::Type getBaseType(mlir::Type t) {
   if (auto nullableT = t.dyn_cast_or_null<mlir::db::NullableType>()) {
      return nullableT.getType();
   }
   return t;
}
bool isIntegerType(mlir::Type type, unsigned int width) {
   auto asStdInt = type.dyn_cast_or_null<mlir::IntegerType>();
   return asStdInt && asStdInt.getWidth() == width;
}
int getIntegerWidth(mlir::Type type, bool isUnSigned) {
   auto asStdInt = type.dyn_cast_or_null<mlir::IntegerType>();
   if (asStdInt && asStdInt.isUnsigned() == isUnSigned) {
      return asStdInt.getWidth();
   }
   return 0;
}
mlir::db::PgNullability mlir::db::combineSqlNullability(mlir::ValueRange values) {
    llvm::SmallVector<mlir::Type> types;
    types.reserve(values.size());
    for (mlir::Value value : values) {
        types.push_back(value.getType());
    }
    return combineSqlNullability(types);
}
mlir::db::PgNullability mlir::db::combineSqlNullability(llvm::ArrayRef<mlir::Type> types) {
    for (mlir::Type type : types) {
        if (mlir::isa<mlir::db::NullableType>(type)) {
            return mlir::db::PgNullability::Maybe;
        }
        if (mlir::db::isPgValueType(type) && mlir::db::getPgNullability(type) == mlir::db::PgNullability::Maybe) {
            return mlir::db::PgNullability::Maybe;
        }
    }
    return mlir::db::PgNullability::Never;
}
static bool isI1Type(mlir::Type type) {
    auto intType = mlir::dyn_cast_or_null<mlir::IntegerType>(type);
    return intType && intType.getWidth() == 1;
}
static bool isPgBoolType(mlir::Type type) {
    return mlir::isa<mlir::db::PgBoolType>(type);
}
static bool isPgBoolWithNullability(mlir::Type type, mlir::db::PgNullability nullability) {
    return isPgBoolType(type) && mlir::db::getPgNullability(type) == nullability;
}
static bool isLegacyNullablePgWrapper(mlir::Type type) {
    auto nullableType = mlir::dyn_cast_or_null<mlir::db::NullableType>(type);
    return nullableType && mlir::db::isPgValueType(nullableType.getType());
}
static mlir::LogicalResult
verifyPgValueResult(mlir::Operation* op, mlir::Type resultType, mlir::db::PgNullability nullability) {
    if (!mlir::db::isPgValueType(resultType)) {
        return op->emitOpError("requires a PostgreSQL semantic result type for PostgreSQL operands");
    }
    if (mlir::db::getPgNullability(resultType) != nullability) {
        return op->emitOpError("result PostgreSQL nullability does not match operand nullability");
    }
    return mlir::success();
}
static mlir::LogicalResult
verifyPgBoolResult(mlir::Operation* op, mlir::Type resultType, mlir::db::PgNullability nullability) {
    if (!isPgBoolType(resultType)) {
        return op->emitOpError("requires a PostgreSQL boolean result type for PostgreSQL operands");
    }
    if (mlir::db::getPgNullability(resultType) != nullability) {
        return op->emitOpError("result PostgreSQL boolean nullability does not match operand nullability");
    }
    return mlir::success();
}
static mlir::LogicalResult verifyLegacyBoolResult(mlir::Operation* op, mlir::Type resultType) {
    if (isI1Type(resultType) || mlir::isa<mlir::db::NullableType>(resultType)) {
        return mlir::success();
    }
    return op->emitOpError("requires i1, legacy nullable, or PostgreSQL boolean result type");
}
static mlir::LogicalResult verifyBinarySqlValueOp(mlir::Operation* op) {
    if (!hasPgValue(op->getOperands())) {
        return mlir::success();
    }
    return verifyPgValueResult(op, op->getResult(0).getType(), mlir::db::combineSqlNullability(op->getOperands()));
}
static mlir::LogicalResult verifyLogicalSqlValueOp(mlir::Operation* op) {
    if (!hasPgValue(op->getOperands())) {
        return verifyLegacyBoolResult(op, op->getResult(0).getType());
    }
    return verifyPgBoolResult(op, op->getResult(0).getType(), mlir::db::combineSqlNullability(op->getOperands()));
}
LogicalResult inferReturnType(MLIRContext* context, std::optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
    if (hasPgValue(operands)) {
        return failure();
    }
   Type baseTypeLeft = getBaseType(operands[0].getType());
   Type baseTypeRight = getBaseType(operands[1].getType());
   Type baseType=baseTypeLeft;
   if(baseTypeLeft.isa<mlir::db::DecimalType>()){
      auto a = baseTypeLeft.dyn_cast_or_null<mlir::db::DecimalType>();
      auto b = baseTypeRight.dyn_cast_or_null<mlir::db::DecimalType>();
      auto hidig = std::max(a.getP() - a.getS(), b.getP() - b.getS());
      auto maxs = std::max(a.getS(), b.getS());
      const auto sump = std::min(hidig + maxs, MAX_NUMERIC_PRECISION);
      const auto sums = std::min(maxs, MAX_NUMERIC_UNCONSTRAINED_SCALE);

      // Addition is super-type of both, with larger precision for carry.
      // TODO: actually add carry precision (+1).
      baseType = mlir::db::DecimalType::get(a.getContext(), sump, sums);
   }
   inferredReturnTypes.push_back(wrapNullableType(context, baseType, operands));
   return success();
}
LogicalResult inferMulReturnType(MLIRContext* context, std::optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
    if (hasPgValue(operands)) {
        return failure();
    }
   Type baseTypeLeft = getBaseType(operands[0].getType());
   Type baseTypeRight = getBaseType(operands[1].getType());
   Type baseType=baseTypeLeft;
   if(baseTypeLeft.isa<mlir::db::DecimalType>()){
      auto a = baseTypeLeft.dyn_cast_or_null<mlir::db::DecimalType>();
      auto b = baseTypeRight.dyn_cast_or_null<mlir::db::DecimalType>();
      auto sump = a.getP() + b.getP();
      auto sums = a.getS() + b.getS();

      sump = std::min(sump, MAX_NUMERIC_PRECISION);
      sums = std::min(sums, MAX_NUMERIC_UNCONSTRAINED_SCALE);

      baseType = mlir::db::DecimalType::get(a.getContext(), sump, sums);
   }
   inferredReturnTypes.push_back(wrapNullableType(context, baseType, operands));
   return success();
}
LogicalResult inferDivReturnType(MLIRContext* context, std::optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
    if (hasPgValue(operands)) {
        return failure();
    }
   Type baseTypeLeft = getBaseType(operands[0].getType());
   Type baseTypeRight = getBaseType(operands[1].getType());
   Type baseType=baseTypeLeft;
   if(baseTypeLeft.isa<mlir::db::DecimalType>()){
      auto leftDecType=baseTypeLeft.dyn_cast_or_null<mlir::db::DecimalType>();
      auto rightDecType=baseTypeRight.dyn_cast_or_null<mlir::db::DecimalType>();

      const auto precision = std::min(std::max(leftDecType.getP(),rightDecType.getP()), MAX_NUMERIC_PRECISION);
      const auto scale = std::min(std::max(leftDecType.getS(),rightDecType.getS()), MAX_NUMERIC_UNCONSTRAINED_SCALE);

      baseType=mlir::db::DecimalType::get(baseType.getContext(), precision, scale);
   }
   inferredReturnTypes.push_back(wrapNullableType(context, baseType, operands));
   return success();
}

::mlir::LogicalResult mlir::db::RuntimeCall::verify() {
   mlir::db::RuntimeCall& runtimeCall=*this;
   auto reg = runtimeCall.getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
   if (!reg->verify(runtimeCall.getFn().str(), runtimeCall.getArgs().getTypes(), runtimeCall.getNumResults() == 1 ? runtimeCall.getResultTypes()[0] : mlir::Type())) {
      runtimeCall->emitError("could not find matching runtime function: ") << runtimeCall.getFn().str();
      return failure();
   }
   return success();
}
bool mlir::db::RuntimeCall::supportsInvalidValues() {
   auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
   if (auto* fn = reg->lookup(this->getFn().str())) {
      return fn->nullHandleType == RuntimeFunction::HandlesInvalidVaues;
   }
   return false;
}
bool mlir::db::RuntimeCall::needsNullWrap() {
   auto reg = getContext()->getLoadedDialect<mlir::db::DBDialect>()->getRuntimeFunctionRegistry();
   if (auto* fn = reg->lookup(this->getFn().str())) {
      return fn->nullHandleType != RuntimeFunction::HandlesNulls;
   }
   return false;
}

LogicalResult mlir::db::PgRowGetOp::verify() {
    auto rowType = mlir::dyn_cast_or_null<mlir::db::PgRowType>(getRow().getType());
    if (!rowType) {
        return emitOpError("requires a !db.pg_row operand");
    }
    uint64_t index = getIndex();
    if (index > std::numeric_limits<uint32_t>::max()) {
        return emitOpError("row field index is outside uint32 range");
    }
    auto field = mlir::db::getPgRowFieldByIndex(rowType, static_cast<uint32_t>(index));
    if (!field) {
        return emitOpError("row field index is outside the row schema");
    }
    if (getResult().getType() != mlir::db::getPgRowFieldType(field)) {
        return emitOpError("result type must match the selected row field type");
    }
    return success();
}

LogicalResult mlir::db::PgRowProjectOp::verify() {
    auto inputType = mlir::dyn_cast_or_null<mlir::db::PgRowType>(getRow().getType());
    if (!inputType) {
        return emitOpError("requires a !db.pg_row operand");
    }
    auto resultType = mlir::dyn_cast_or_null<mlir::db::PgRowType>(getResult().getType());
    if (!resultType) {
        return emitOpError("requires a !db.pg_row result");
    }
    if (resultType.getSchema() != getSchema()) {
        return emitOpError("result row schema must match the project schema attribute");
    }
    (void)inputType;
    return success();
}

LogicalResult mlir::db::PgEmitRowOp::verify() {
    auto fields = getSchema().getFields();
    if (getValues().size() != fields.size()) {
        return emitOpError("value count must match row schema field count");
    }
    for (auto [index, value] : llvm::enumerate(getValues())) {
        if (value.getType() != fields[index].getType()) {
            return emitOpError("value type must match row schema field type");
        }
    }
    return success();
}

bool mlir::db::CmpOp::supportsInvalidValues() {
   auto type = getBaseType(getLeft().getType());
   if (type.isa<db::StringType, db::DecimalType>()) {
       return false;
   }
   return true;
}
bool mlir::db::CastOp::supportsInvalidValues() {
    if (getBaseType(getResult().getType()).isa<db::StringType, db::DecimalType>()
        || getBaseType(getVal().getType()).isa<db::StringType, db::DecimalType>())
    {
        return false;
    }
   return true;
}
static bool binaryOpSupportsInvalidValues(mlir::Type leftType) {
    return !getBaseType(leftType).isa<mlir::db::DecimalType>();
}
bool mlir::db::AddOp::supportsInvalidValues() {
    return binaryOpSupportsInvalidValues(getLeft().getType());
}
bool mlir::db::SubOp::supportsInvalidValues() {
    return binaryOpSupportsInvalidValues(getLeft().getType());
}
bool mlir::db::MulOp::supportsInvalidValues() {
    return binaryOpSupportsInvalidValues(getLeft().getType());
}
bool mlir::db::DivOp::supportsInvalidValues() {
    return binaryOpSupportsInvalidValues(getLeft().getType());
}
bool mlir::db::ModOp::supportsInvalidValues() {
    return binaryOpSupportsInvalidValues(getLeft().getType());
}

LogicalResult mlir::db::ConstantOp::verify() {
    mlir::Type resultType = getResult().getType();
    if (mlir::db::isPgValueType(resultType) && mlir::db::getPgNullability(resultType) != mlir::db::PgNullability::Never)
    {
        return emitOpError("PostgreSQL constants must use a non-null PostgreSQL result type");
    }
    return success();
}

LogicalResult mlir::db::NullOp::verify() {
    mlir::Type resultType = getRes().getType();
    if (auto nullableType = mlir::dyn_cast_or_null<mlir::db::NullableType>(resultType)) {
        if (mlir::db::isPgValueType(nullableType.getType())) {
            return emitOpError("legacy nullable cannot wrap PostgreSQL semantic types");
        }
        return success();
    }
    if (!mlir::db::isPgValueType(resultType)) {
        return emitOpError("requires a legacy nullable or nullable PostgreSQL result type");
    }
    if (mlir::db::getPgNullability(resultType) != mlir::db::PgNullability::Maybe) {
        return emitOpError("PostgreSQL nulls must use a nullable PostgreSQL result type");
    }
    return success();
}

LogicalResult mlir::db::AsNullableOp::verify() {
    mlir::Type resultType = getRes().getType();
    if (auto nullableType = mlir::dyn_cast_or_null<mlir::db::NullableType>(resultType)) {
        if (mlir::db::isPgValueType(nullableType.getType())) {
            return emitOpError("legacy nullable cannot wrap PostgreSQL semantic types");
        }
        return success();
    }

    if (!mlir::db::isPgValueType(resultType)) {
        return emitOpError("requires a legacy nullable or nullable PostgreSQL result type");
    }
    if (mlir::db::getPgNullability(resultType) != mlir::db::PgNullability::Maybe) {
        return emitOpError("PostgreSQL result must be nullable");
    }

    mlir::Type valueType = getVal().getType();
    if (!mlir::db::isPgValueType(valueType)) {
        return emitOpError("requires a PostgreSQL operand for a PostgreSQL result type");
    }
    if (mlir::db::withPgNullability(valueType, mlir::db::PgNullability::Maybe) != resultType) {
        return emitOpError("PostgreSQL result type must match the operand type except for nullability");
    }
    return success();
}

LogicalResult mlir::db::IsNullOp::verify() {
    mlir::Type valueType = getVal().getType();
    mlir::Type resultType = getResult().getType();
    if (mlir::db::isPgValueType(valueType)) {
        if (!isPgBoolWithNullability(resultType, mlir::db::PgNullability::Never)) {
            return emitOpError("over a PostgreSQL value requires non-null !db.pg_bool result type");
        }
        return success();
    }
    return verifyLegacyBoolResult(*this, resultType);
}

LogicalResult mlir::db::CastOp::verify() {
    mlir::Type valueType = getVal().getType();
    if (isLegacyNullablePgWrapper(valueType) || isLegacyNullablePgWrapper(getResult().getType())) {
        return emitOpError("legacy nullable cannot wrap PostgreSQL semantic types");
    }
    if (!mlir::db::isPgValueType(valueType)) {
        return success();
    }
    return verifyPgValueResult(*this, getResult().getType(), mlir::db::getPgNullability(valueType));
}

LogicalResult mlir::db::AddOp::verify() {
    return verifyBinarySqlValueOp(*this);
}
LogicalResult mlir::db::SubOp::verify() {
    return verifyBinarySqlValueOp(*this);
}
LogicalResult mlir::db::MulOp::verify() {
    return verifyBinarySqlValueOp(*this);
}
LogicalResult mlir::db::DivOp::verify() {
    return verifyBinarySqlValueOp(*this);
}
LogicalResult mlir::db::ModOp::verify() {
    return verifyBinarySqlValueOp(*this);
}

LogicalResult mlir::db::CmpOp::verify() {
    return verifyLogicalSqlValueOp(*this);
}
LogicalResult mlir::db::BetweenOp::verify() {
    return verifyLogicalSqlValueOp(*this);
}
LogicalResult mlir::db::OneOfOp::verify() {
    return verifyLogicalSqlValueOp(*this);
}
LogicalResult mlir::db::AndOp::verify() {
    return verifyLogicalSqlValueOp(*this);
}
LogicalResult mlir::db::OrOp::verify() {
    return verifyLogicalSqlValueOp(*this);
}
LogicalResult mlir::db::NotOp::verify() {
    return verifyLogicalSqlValueOp(*this);
}

LogicalResult mlir::db::OrOp::canonicalize(mlir::db::OrOp orOp, mlir::PatternRewriter& rewriter) {
   llvm::SmallDenseMap<mlir::Value, size_t> usage;
   for (auto val : orOp.getVals()) {
      if (!val.getDefiningOp()) return failure();
      if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(val.getDefiningOp())) {
         llvm::SmallPtrSet<mlir::Value, 4> alreadyUsed;
         for (auto andOperand : andOp.getVals()) {
            if (!alreadyUsed.contains(andOperand)) {
               usage[andOperand]++;
               alreadyUsed.insert(andOperand);
            }
         }
      } else {
         return failure();
      }
   }
   size_t totalAnds = orOp.getVals().size();
   llvm::SmallPtrSet<mlir::Value, 4> extracted;
   std::vector<mlir::Value> newOrOperands;
   for (auto val : orOp.getVals()) {
      if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(val.getDefiningOp())) {
         std::vector<mlir::Value> keep;
         for (auto andOperand : andOp.getVals()) {
            if (usage[andOperand] == totalAnds) {
               extracted.insert(andOperand);
            } else {
               keep.push_back(andOperand);
            }
         }
         if (keep.size() != andOp.getVals().size()) {
            if (keep.size()) {
               newOrOperands.push_back(rewriter.create<mlir::db::AndOp>(andOp->getLoc(), keep));
            }
         } else {
            newOrOperands.push_back(andOp);
         }
      }
   }
   std::vector<Value> extractedAsVec;
   extractedAsVec.insert(extractedAsVec.end(), extracted.begin(), extracted.end());
   if (!extracted.empty()) {
      if (newOrOperands.size() == 1) {
         extractedAsVec.push_back(newOrOperands[0]);
      } else if (newOrOperands.size() > 1) {
         Value newOrOp = rewriter.create<mlir::db::OrOp>(orOp->getLoc(), newOrOperands);
         extractedAsVec.push_back(newOrOp);
      }
      rewriter.replaceOpWithNewOp<mlir::db::AndOp>(orOp, extractedAsVec);
      return success();
   } else if (newOrOperands.size() == 1) {
      rewriter.replaceOp(orOp, newOrOperands[0]);
   }
   return failure();
}
LogicalResult mlir::db::AndOp::canonicalize(mlir::db::AndOp andOp, mlir::PatternRewriter& rewriter) {
   llvm::DenseSet<mlir::Value> rawValues;
   llvm::DenseMap<mlir::Value, std::vector<mlir::db::CmpOp>> cmps;
   std::queue<mlir::Value> queue;
   queue.push(andOp);
   while (!queue.empty()) {
      auto current = queue.front();
      queue.pop();
      if (auto* definingOp = current.getDefiningOp()) {
         if (auto nestedAnd = mlir::dyn_cast_or_null<mlir::db::AndOp>(definingOp)) {
            for (auto v : nestedAnd.getVals()) {
               queue.push(v);
            }
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(definingOp)) {
            cmps[cmpOp.getLeft()].push_back(cmpOp);
            cmps[cmpOp.getRight()].push_back(cmpOp);
            rawValues.insert(current);
         } else {
            rawValues.insert(current);
         }
      } else {
         rawValues.insert(current);
      }
   }
   for (auto m : cmps) {
      mlir::Value lower, upper;
      mlir::db::CmpOp lowerCmp, upperCmp;
      mlir::Value current = m.getFirst();
      if (auto* definingOp = current.getDefiningOp()) {
         if (mlir::isa<mlir::db::ConstantOp>(definingOp)) {
            continue;
         }
      }
      for (auto cmp : m.second) {
         if (!rawValues.contains(cmp)) continue;
         switch (cmp.getPredicate()) {
            case DBCmpPredicate::lt:
            case DBCmpPredicate::lte:
               if (cmp.getLeft() == current) {
                  upper = cmp.getRight();
                  upperCmp = cmp;
               } else {
                  lower = cmp.getLeft();
                  lowerCmp = cmp;
               }
               break;
            case DBCmpPredicate::gt:
            case DBCmpPredicate::gte:
               if (cmp.getLeft() == current) {
                  lower = cmp.getRight();
                  lowerCmp = cmp;
               } else {
                  upper = cmp.getLeft();
                  upperCmp = cmp;
               }
               break;
            default: break;
         }
      }
      if (lower && upper && lower.getDefiningOp() && upper.getDefiningOp() && mlir::isa<mlir::db::ConstantOp>(lower.getDefiningOp()) && mlir::isa<mlir::db::ConstantOp>(upper.getDefiningOp())) {
         auto lowerInclusive = lowerCmp.getPredicate() == DBCmpPredicate::gte || lowerCmp.getPredicate() == DBCmpPredicate::lte;
         auto upperInclusive = upperCmp.getPredicate() == DBCmpPredicate::gte || upperCmp.getPredicate() == DBCmpPredicate::lte;
         mlir::Value between = rewriter.create<mlir::db::BetweenOp>(lowerCmp->getLoc(), current, lower, upper, lowerInclusive, upperInclusive);
         rawValues.erase(lowerCmp);
         rawValues.erase(upperCmp);
         rawValues.insert(between);
      }
   }
   if(rawValues.size()==1){
      rewriter.replaceOp(andOp,*rawValues.begin());
      return success();
   }
   if (rawValues.size() != andOp.getVals().size()) {
      rewriter.replaceOpWithNewOp<mlir::db::AndOp>(andOp, std::vector<mlir::Value>(rawValues.begin(), rawValues.end()));
      return success();
   }
   return failure();
}
#define GET_OP_CLASSES
#include "lingodb/mlir/Dialect/DB/IR/DBOps.cpp.inc"
#include "lingodb/mlir/Dialect/DB/IR/DBOpsInterfaces.cpp.inc"
