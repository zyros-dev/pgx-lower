#include "dialects/db/DBOps.h"
#include "dialects/db/DBDialect.h"
// #include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h" // TODO Phase 5: Port if needed
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>

// #include "lingodb/compiler/mlir-support/parsing.h" // TODO Phase 5: Port if needed

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;
using namespace pgx_lower::compiler::dialect;

bool pgx_lower::compiler::dialect::db::CmpOp::isEqualityPred(bool nullsAreEqual) { return getPredicate() == db::DBCmpPredicate::eq || (nullsAreEqual ? (getPredicate() == DBCmpPredicate::isa) : false); }
bool pgx_lower::compiler::dialect::db::CmpOp::isUnequalityPred() { return getPredicate() == db::DBCmpPredicate::neq; }
bool pgx_lower::compiler::dialect::db::CmpOp::isLessPred(bool eq) { return getPredicate() == (eq ? db::DBCmpPredicate::lte : db::DBCmpPredicate::lt); }
bool pgx_lower::compiler::dialect::db::CmpOp::isGreaterPred(bool eq) { return getPredicate() == (eq ? db::DBCmpPredicate::gte : db::DBCmpPredicate::gt); }
mlir::Type getBaseType(mlir::Type t) {
   if (auto nullableT = mlir::dyn_cast_or_null<db::NullableType>(t)) {
      return nullableT.getType();
   }
   return t;
}
Type wrapNullableType(MLIRContext* context, Type type, ValueRange values) {
   if (llvm::any_of(values, [](Value v) { return mlir::isa<db::NullableType>(v.getType()); })) {
      return db::NullableType::get(type);
   }
   return type;
}

// Helper functions for type inference - these need to be in global scope for TableGen
LogicalResult inferReturnType(MLIRContext* context, std::optional<Location> location,
                             ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   // Basic implementation - just use the first operand's type
   if (operands.empty()) return failure();
   inferredReturnTypes.push_back(wrapNullableType(context, getBaseType(operands[0].getType()), operands));
   return success();
}

LogicalResult inferMulReturnType(MLIRContext* context, std::optional<Location> location,
                                  ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   // TODO Phase 6: Implement proper multiplication type inference
   return inferReturnType(context, location, operands, inferredReturnTypes);
}

LogicalResult inferDivReturnType(MLIRContext* context, std::optional<Location> location,
                                  ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   // TODO Phase 6: Implement proper division type inference (float result)
   return inferReturnType(context, location, operands, inferredReturnTypes);
}

LogicalResult inferRemReturnType(MLIRContext* context, std::optional<Location> location,
                                  ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   // TODO Phase 6: Implement proper remainder type inference
   return inferReturnType(context, location, operands, inferredReturnTypes);
}

LogicalResult inferArithmeticReturnType(MLIRContext* context, std::optional<Location> location, ValueRange operands, SmallVectorImpl<Type>& inferredReturnTypes) {
   Type baseTypeLeft = getBaseType(operands[0].getType());
   Type baseTypeRight = getBaseType(operands[1].getType());
   Type baseType = baseTypeLeft;
   if (mlir::isa<db::DecimalType>(baseTypeLeft)) {
      auto a = mlir::cast<db::DecimalType>(baseTypeLeft);
      auto b = mlir::cast<db::DecimalType>(baseTypeRight);
      auto hidig = std::max(a.getP() - a.getS(), b.getP() - b.getS());
      auto maxs = std::max(a.getS(), b.getS());
      // Addition is super-type of both, with larger precision for carry.
      // TODO Phase 6: actually add carry precision (+1) for arithmetic operations.
      baseType = db::DecimalType::get(a.getContext(), hidig + maxs, maxs);
   }
   inferredReturnTypes.push_back(wrapNullableType(context, baseType, operands));
   return success();
}

bool isIntegerType(mlir::Type type, unsigned int width) {
   auto asStdInt = mlir::dyn_cast_or_null<mlir::IntegerType>(type);
   return asStdInt && asStdInt.getWidth() == width;
}
int getIntegerWidth(mlir::Type type, bool isUnSigned) {
   auto asStdInt = mlir::dyn_cast_or_null<mlir::IntegerType>(type);
   if (asStdInt && asStdInt.isUnsigned() == isUnSigned) {
      return asStdInt.getWidth();
   }
   return 0;
}
namespace {

mlir::Type getAdaptedDecimalTypeAfterMulDiv(mlir::MLIRContext* context, int precision, int scale) {
   int beforeComma = precision - scale;
   if (beforeComma > 32 && scale > 6) {
      return db::DecimalType::get(context, 38, 6);
   }
   if (beforeComma > 32 && scale <= 6) {
      return db::DecimalType::get(context, 38, scale);
   }
   return db::DecimalType::get(context, std::min(precision, 38), std::min(scale, 38 - beforeComma));
}
} // end anonymous namespace

// PostgreSQL: Simple ConstantOp fold implementation without Arrow conversion
mlir::OpFoldResult pgx_lower::compiler::dialect::db::ConstantOp::fold(db::ConstantOp::FoldAdaptor adaptor) {
   // For constants, just return the value attribute as-is since it's already properly typed
   return getValue();
}

mlir::OpFoldResult pgx_lower::compiler::dialect::db::AddOp::fold(db::AddOp::FoldAdaptor adaptor) {
   auto left = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getLeft());
   auto right = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getRight());
   if (left && right && left.getType() == right.getType()) {
      return IntegerAttr::get(left.getType(), left.getValue() + right.getValue());
   }
   return {};
}
mlir::OpFoldResult pgx_lower::compiler::dialect::db::SubOp::fold(db::SubOp::FoldAdaptor adaptor) {
   auto left = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getLeft());
   auto right = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getRight());
   if (left && right && left.getType() == right.getType()) {
      return IntegerAttr::get(left.getType(), left.getValue() - right.getValue());
   }
   return {};
}

// PostgreSQL: Simple CastOp fold implementation without Arrow support
mlir::OpFoldResult pgx_lower::compiler::dialect::db::CastOp::fold(db::CastOp::FoldAdaptor adaptor) {
   auto scalarSourceType = getVal().getType();
   auto scalarTargetType = getType();
   
   // Skip string casts and nullable types for now
   if (mlir::isa<db::StringType>(scalarSourceType) || mlir::isa<db::StringType>(scalarTargetType)) return {};
   if (mlir::isa<db::NullableType>(scalarSourceType)) return {};
   
   // If types are the same, no cast needed
   if (scalarSourceType == scalarTargetType) {
      return adaptor.getVal();
   }
   
   // TODO Phase 5: Implement proper type conversions without Arrow support
   // For now, skip folding complex casts to avoid Arrow dependencies
   return {};
}

// PostgreSQL: Simple RuntimeCall fold implementation without registry
mlir::LogicalResult pgx_lower::compiler::dialect::db::RuntimeCall::fold(db::RuntimeCall::FoldAdaptor adaptor, llvm::SmallVectorImpl<mlir::OpFoldResult>& results) {
   // TODO Phase 5: Implement proper runtime function folding
   // For now, skip folding runtime calls to avoid registry dependencies
   return failure();
}
// PostgreSQL: Simple RuntimeCall verify implementation without registry
mlir::LogicalResult pgx_lower::compiler::dialect::db::RuntimeCall::verify() {
   // TODO Phase 5: Implement proper runtime function verification
   // For now, just accept all runtime calls
   return success();
}

// PostgreSQL: Simple RuntimeCall method implementations without registry
bool pgx_lower::compiler::dialect::db::RuntimeCall::supportsInvalidValues() {
   // TODO Phase 5: Implement proper runtime function analysis
   // For now, assume most runtime calls don't support invalid values
   return false;
}

bool pgx_lower::compiler::dialect::db::RuntimeCall::needsNullWrap() {
   // TODO Phase 5: Implement proper null handling analysis
   // For now, assume most runtime calls need null wrapping
   return true;
}

bool pgx_lower::compiler::dialect::db::CmpOp::supportsInvalidValues() {
   auto type = getBaseType(getLeft().getType());
   if (mlir::isa<db::StringType>(type)) {
      return false;
   }
   return true;
}
bool pgx_lower::compiler::dialect::db::CastOp::supportsInvalidValues() {
   if (mlir::isa<db::StringType>(getBaseType(getResult().getType())) || mlir::isa<db::StringType>(getBaseType(getVal().getType()))) {
      return false;
   }
   return true;
}

LogicalResult db::OrOp::canonicalize(db::OrOp orOp, mlir::PatternRewriter& rewriter) {
   llvm::SmallDenseMap<mlir::Value, size_t> usage;
   for (auto val : orOp.getVals()) {
      if (!val.getDefiningOp()) return failure();
      if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(val.getDefiningOp())) {
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
      if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(val.getDefiningOp())) {
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
               newOrOperands.push_back(rewriter.create<db::AndOp>(andOp->getLoc(), mlir::ValueRange(keep), llvm::ArrayRef<mlir::NamedAttribute>{}));
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
         Value newOrOp = rewriter.create<db::OrOp>(orOp->getLoc(), newOrOperands);
         extractedAsVec.push_back(newOrOp);
      }
      rewriter.replaceOpWithNewOp<db::AndOp>(orOp, mlir::ValueRange(extractedAsVec), llvm::ArrayRef<mlir::NamedAttribute>{});
      return success();
   } else if (newOrOperands.size() == 1) {
      rewriter.replaceOp(orOp, newOrOperands[0]);
   }
   return failure();
}
OpFoldResult pgx_lower::compiler::dialect::db::IsNullOp::fold(FoldAdaptor adaptor) {
   auto nullableVal = getVal();
   if (!mlir::isa<db::NullableType>(nullableVal.getType())) {
      return mlir::BoolAttr::get(getContext(), false);
   } else if (auto asNullableOp = mlir::dyn_cast_or_null<db::AsNullableOp>(nullableVal.getDefiningOp())) {
      if (asNullableOp.getNull()) {
         return asNullableOp.getNull();
      } else {
         return mlir::BoolAttr::get(getContext(), false);
      }
   }
   return {};
}
OpFoldResult pgx_lower::compiler::dialect::db::NullableGetVal::fold(FoldAdaptor adaptor) {
   auto nullableVal = getVal();
   if (auto asNullableOp = mlir::dyn_cast_or_null<db::AsNullableOp>(nullableVal.getDefiningOp())) {
      return asNullableOp.getVal();
   }
   return {};
}
LogicalResult db::AndOp::canonicalize(db::AndOp andOp, mlir::PatternRewriter& rewriter) {
   llvm::DenseSet<mlir::Value> rawValues;
   llvm::DenseMap<mlir::Value, std::vector<db::CmpOp>> cmps;
   std::queue<mlir::Value> queue;
   queue.push(andOp);
   while (!queue.empty()) {
      auto current = queue.front();
      queue.pop();
      if (auto* definingOp = current.getDefiningOp()) {
         if (auto nestedAnd = mlir::dyn_cast_or_null<db::AndOp>(definingOp)) {
            for (auto v : nestedAnd.getVals()) {
               queue.push(v);
            }
         } else if (auto cmpOp = mlir::dyn_cast_or_null<db::CmpOp>(definingOp)) {
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
      db::CmpOp lowerCmp, upperCmp;
      mlir::Value current = m.getFirst();
      if (auto* definingOp = current.getDefiningOp()) {
         if (mlir::isa<db::ConstantOp>(definingOp)) {
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
      if (lower && upper && lower.getDefiningOp() && upper.getDefiningOp() && mlir::isa<db::ConstantOp>(lower.getDefiningOp()) && mlir::isa<db::ConstantOp>(upper.getDefiningOp())) {
         auto lowerInclusive = lowerCmp.getPredicate() == DBCmpPredicate::gte || lowerCmp.getPredicate() == DBCmpPredicate::lte;
         auto upperInclusive = upperCmp.getPredicate() == DBCmpPredicate::gte || upperCmp.getPredicate() == DBCmpPredicate::lte;
         mlir::Value between = rewriter.create<db::BetweenOp>(lowerCmp->getLoc(), current, lower, upper, lowerInclusive, upperInclusive);
         rawValues.erase(lowerCmp);
         rawValues.erase(upperCmp);
         rawValues.insert(between);
      }
   }
   if (rawValues.size() == 1) {
      rewriter.replaceOp(andOp, *rawValues.begin());
      return success();
   }
   if (rawValues.size() != andOp.getVals().size()) {
      std::vector<mlir::Value> valuesVec(rawValues.begin(), rawValues.end());
      rewriter.replaceOpWithNewOp<db::AndOp>(andOp, mlir::ValueRange(valuesVec), llvm::ArrayRef<mlir::NamedAttribute>{});
      return success();
   }
   return failure();
}
#define GET_OP_CLASSES
#include "DBOps.cpp.inc"
#include "DBOpsInterfaces.cpp.inc"
#include "DBOpsEnums.cpp.inc"
