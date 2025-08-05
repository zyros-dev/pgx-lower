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
    printer.printAttributeWithoutType(getTableIdentifierAttr());
    printer << " ";
    printer.printAttribute(getMeta());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"table_identifier", "meta"});
    printer << " ";
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

ParseResult BaseTableOp::parse(OpAsmParser& parser, OperationState& result) {
    StringAttr tableId;
    TableMetaDataAttr meta;
    Type resultType;
    
    if (parser.parseAttribute(tableId) ||
        parser.parseAttribute(meta) ||
        parser.parseArrow() ||
        parser.parseType(resultType)) {
        return failure();
    }
    
    result.addAttribute("table_identifier", tableId);
    result.addAttribute("meta", meta);
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
    printer << " ";
    printer.printAttribute(getCols());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"cols"});
}

ParseResult ProjectionOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand rel;
    ArrayAttr cols;
    Type relType, resultType;
    
    if (parser.parseOperand(rel) ||
        parser.parseAttribute(cols) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(relType)) {
        return failure();
    }
    
    result.addAttribute("cols", cols);
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
    printer << " computes ";
    printer.printAttribute(getComputedCols());
    printer << " ";
    printer.printRegion(getComputation(), /*printEntryBlockArgs=*/false);
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"computed_cols"});
}

ParseResult MapOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand rel;
    ArrayAttr computedCols;
    Type relType, resultType;
    
    if (parser.parseOperand(rel) ||
        parser.parseKeyword("computes") ||
        parser.parseAttribute(computedCols)) {
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
    
    result.addAttribute("computed_cols", computedCols);
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
    printer << " ";
    printer.printAttribute(getCols());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"cols"});
}

ParseResult MaterializeOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand rel;
    ArrayAttr cols;
    Type relType, resultType;
    
    if (parser.parseOperand(rel) ||
        parser.parseAttribute(cols) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(relType)) {
        return failure();
    }
    
    result.addAttribute("cols", cols);
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
    printer.printOperand(getTuple());
    printer << " ";
    printer.printAttribute(getAttr());
    printer << " -> ";
    printer.printType(getRes().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"attr"});
}

ParseResult GetColumnOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand tuple;
    ColumnRefAttr attr;
    Type tupleType, resultType;
    
    if (parser.parseOperand(tuple) ||
        parser.parseAttribute(attr) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(tupleType)) {
        return failure();
    }
    
    result.addAttribute("attr", attr);
    result.addTypes(resultType);
    
    if (parser.resolveOperand(tuple, tupleType, result.operands)) {
        return failure();
    }
    
    return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"