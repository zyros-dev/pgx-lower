#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "execution/logging.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace pgx::mlir::relalg;

//===----------------------------------------------------------------------===//
// BaseTableOp
//===----------------------------------------------------------------------===//

void BaseTableOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printAttributeWithoutType(getTableNameAttr());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"table_name"});
    printer << " ";
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

ParseResult BaseTableOp::parse(OpAsmParser& parser, OperationState& result) {
    StringAttr tableName;
    Type resultType;
    
    if (parser.parseAttribute(tableName) ||
        parser.parseArrow() ||
        parser.parseType(resultType)) {
        return failure();
    }
    
    result.addAttribute("table_name", tableName);
    result.addTypes(resultType);
    
    Region* body = result.addRegion();
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{})) {
        return failure();
    }
    
    BaseTableOp::ensureTerminator(*body, parser.getBuilder(), result.location);
    return success();
}

//===----------------------------------------------------------------------===//
// ProjectionOp
//===----------------------------------------------------------------------===//

void ProjectionOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getRel());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ProjectionOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand rel;
    Type relType, resultType;
    
    if (parser.parseOperand(rel) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(relType)) {
        return failure();
    }
    
    result.addTypes(resultType);
    
    if (parser.resolveOperand(rel, relType, result.operands)) {
        return failure();
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// SelectionOp  
//===----------------------------------------------------------------------===//

void SelectionOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getRel());
    printer << " ";
    printer.printRegion(getPredicate(), /*printEntryBlockArgs=*/false);
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult SelectionOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand rel;
    Type relType, resultType;
    
    if (parser.parseOperand(rel)) {
        return failure();
    }
    
    Region* predicate = result.addRegion();
    if (parser.parseRegion(*predicate, /*arguments=*/{}, /*argTypes=*/{})) {
        return failure();
    }
    
    if (parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(relType)) {
        return failure();
    }
    
    result.addTypes(resultType);
    
    if (parser.resolveOperand(rel, relType, result.operands)) {
        return failure();
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

void MapOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getRel());
    printer << " ";
    printer.printRegion(getComputation(), /*printEntryBlockArgs=*/false);
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult MapOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand rel;
    Type relType, resultType;
    
    if (parser.parseOperand(rel)) {
        return failure();
    }
    
    Region* computation = result.addRegion();
    if (parser.parseRegion(*computation, /*arguments=*/{}, /*argTypes=*/{})) {
        return failure();
    }
    
    if (parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(relType)) {
        return failure();
    }
    
    result.addTypes(resultType);
    
    if (parser.resolveOperand(rel, relType, result.operands)) {
        return failure();
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// MaterializeOp
//===----------------------------------------------------------------------===//

void MaterializeOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getRel());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult MaterializeOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand rel;
    Type relType, resultType;
    
    if (parser.parseOperand(rel) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(relType)) {
        return failure();
    }
    
    result.addTypes(resultType);
    
    if (parser.resolveOperand(rel, relType, result.operands)) {
        return failure();
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// GetColumnOp
//===----------------------------------------------------------------------===//

void GetColumnOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printAttributeWithoutType(getColumnNameAttr());
    printer << " ";
    printer.printOperand(getTuple());
    printer << " -> ";
    printer.printType(getRes().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"column_name"});
}

ParseResult GetColumnOp::parse(OpAsmParser& parser, OperationState& result) {
    StringAttr columnName;
    OpAsmParser::UnresolvedOperand tuple;
    Type tupleType, resultType;
    
    if (parser.parseAttribute(columnName) ||
        parser.parseOperand(tuple) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(tupleType)) {
        return failure();
    }
    
    result.addAttribute("column_name", columnName);
    result.addTypes(resultType);
    
    if (parser.resolveOperand(tuple, tupleType, result.operands)) {
        return failure();
    }
    
    return success();
}

// Properties system disabled - using traditional attributes

#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"