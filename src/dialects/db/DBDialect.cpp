//===- DBDialect.cpp - Database dialect implementation -----------*- C++ -*-===//
//
// Database dialect implementation
//
//===----------------------------------------------------------------------===//

#include "dialects/db/DBDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::db;

#include "DBDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Database dialect initialization
//===----------------------------------------------------------------------===//

void DBDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "DBOps.cpp.inc"
    >();
    
    addTypes<
#define GET_TYPEDEF_LIST
#include "DBTypes.cpp.inc"
    >();
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "DBTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Binary operation type inference
//===----------------------------------------------------------------------===//

static Type wrapNullableType(MLIRContext *context, Type baseType, bool isNullable) {
    if (isNullable && !isa<NullableType>(baseType)) {
        return NullableType::get(context, baseType);
    }
    return baseType;
}

static LogicalResult inferBinaryOpType(
    MLIRContext *context,
    ValueRange operands,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    
    if (operands.size() != 2) {
        return failure();
    }
    
    Type leftType = operands[0].getType();
    Type rightType = operands[1].getType();
    
    // Check if either operand is nullable
    bool isNullable = isa<NullableType>(leftType) || isa<NullableType>(rightType);
    
    // Extract base types
    if (auto nullableLeft = dyn_cast<NullableType>(leftType)) {
        leftType = nullableLeft.getValueType();
    }
    if (auto nullableRight = dyn_cast<NullableType>(rightType)) {
        rightType = nullableRight.getValueType();
    }
    
    // For now, assume both operands have the same base type
    // In a full implementation, we'd handle type promotion
    Type resultBaseType = leftType;
    
    // Wrap in nullable if needed
    Type resultType = wrapNullableType(context, resultBaseType, isNullable);
    inferredReturnTypes.push_back(resultType);
    
    return success();
}

//===----------------------------------------------------------------------===//
// Operation definitions
//===----------------------------------------------------------------------===//

// Binary operation type inference implementations
LogicalResult AddOp::inferReturnType(
    MLIRContext *context,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    return inferBinaryOpType(context, operands, inferredReturnTypes);
}

LogicalResult SubOp::inferReturnType(
    MLIRContext *context,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    return inferBinaryOpType(context, operands, inferredReturnTypes);
}

LogicalResult MulOp::inferReturnType(
    MLIRContext *context,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    return inferBinaryOpType(context, operands, inferredReturnTypes);
}

LogicalResult DivOp::inferReturnType(
    MLIRContext *context,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    return inferBinaryOpType(context, operands, inferredReturnTypes);
}

LogicalResult ModOp::inferReturnType(
    MLIRContext *context,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
    return inferBinaryOpType(context, operands, inferredReturnTypes);
}

// Constant folding
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
    return getValue();
}

// Custom assembly format for ConstantOp
ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
    Attribute valueAttr;
    Type type;
    
    if (parser.parseAttribute(valueAttr, "value", result.attributes) ||
        parser.parseColonType(type))
        return failure();
    
    result.addTypes(type);
    return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
    p << " ";
    p.printAttribute(getValue());
    p << " : ";
    p.printType(getType());
}

#define GET_OP_CLASSES
#include "DBOps.cpp.inc"