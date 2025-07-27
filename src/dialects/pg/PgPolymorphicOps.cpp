//===----------------------------------------------------------------------===//
// PostgreSQL Polymorphic Operations Implementation
//===----------------------------------------------------------------------===//

#include "dialects/pg/PgDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::pg;

//===----------------------------------------------------------------------===//
// GetFieldOp
//===----------------------------------------------------------------------===//

LogicalResult GetFieldOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    
    // Get the field type OID from attributes
    auto fieldTypeOid = attributes.get("field_type_oid").cast<IntegerAttr>().getInt();
    
    Type fieldType;
    switch (fieldTypeOid) {
        case 20:  // INT8OID
            fieldType = IntegerType::get(context, 64);
            break;
        case 23:  // INT4OID
            fieldType = IntegerType::get(context, 32);
            break;
        case 700: // FLOAT4OID
            fieldType = Float32Type::get(context);
            break;
        case 701: // FLOAT8OID
            fieldType = Float64Type::get(context);
            break;
        case 1700: // NUMERICOID
            fieldType = Float64Type::get(context); // For now, treat as float64
            break;
        case 25:   // TEXTOID
        case 1043: // VARCHAROID
            fieldType = IntegerType::get(context, 64); // Pointer to text
            break;
        case 16:   // BOOLOID
            fieldType = IntegerType::get(context, 1);
            break;
        default:
            // Default to i32 for unknown types
            fieldType = IntegerType::get(context, 32);
            break;
    }
    
    inferredReturnTypes.push_back(fieldType);
    inferredReturnTypes.push_back(IntegerType::get(context, 1)); // is_null
    
    return success();
}

//===----------------------------------------------------------------------===//
// Binary Operations Type Inference
//===----------------------------------------------------------------------===//

static Type inferBinaryOpType(Type lhs, Type rhs) {
    // If both are the same type, return that type
    if (lhs == rhs) {
        return lhs;
    }
    
    // Handle mixed integer/float operations
    if (lhs.isa<FloatType>() || rhs.isa<FloatType>()) {
        // Promote to float
        if (lhs.isa<Float64Type>() || rhs.isa<Float64Type>()) {
            return Float64Type::get(lhs.getContext());
        }
        return Float32Type::get(lhs.getContext());
    }
    
    // Handle mixed integer sizes
    if (auto lhsInt = lhs.dyn_cast<IntegerType>()) {
        if (auto rhsInt = rhs.dyn_cast<IntegerType>()) {
            // Promote to larger integer
            unsigned width = std::max(lhsInt.getWidth(), rhsInt.getWidth());
            return IntegerType::get(lhs.getContext(), width);
        }
    }
    
    // Default to lhs type
    return lhs;
}

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    
    if (operands.size() != 2) {
        return failure();
    }
    
    Type resultType = inferBinaryOpType(operands[0].getType(), operands[1].getType());
    inferredReturnTypes.push_back(resultType);
    
    return success();
}

LogicalResult SubOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    
    if (operands.size() != 2) {
        return failure();
    }
    
    Type resultType = inferBinaryOpType(operands[0].getType(), operands[1].getType());
    inferredReturnTypes.push_back(resultType);
    
    return success();
}

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    
    if (operands.size() != 2) {
        return failure();
    }
    
    Type resultType = inferBinaryOpType(operands[0].getType(), operands[1].getType());
    inferredReturnTypes.push_back(resultType);
    
    return success();
}

LogicalResult DivOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    
    if (operands.size() != 2) {
        return failure();
    }
    
    // Division always returns float
    Type resultType;
    if (operands[0].getType().isa<Float64Type>() || 
        operands[1].getType().isa<Float64Type>()) {
        resultType = Float64Type::get(context);
    } else {
        resultType = Float32Type::get(context);
    }
    
    inferredReturnTypes.push_back(resultType);
    
    return success();
}

LogicalResult MinOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    
    if (operands.size() != 2) {
        return failure();
    }
    
    Type resultType = inferBinaryOpType(operands[0].getType(), operands[1].getType());
    inferredReturnTypes.push_back(resultType);
    
    return success();
}

LogicalResult MaxOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    
    if (operands.size() != 2) {
        return failure();
    }
    
    Type resultType = inferBinaryOpType(operands[0].getType(), operands[1].getType());
    inferredReturnTypes.push_back(resultType);
    
    return success();
}