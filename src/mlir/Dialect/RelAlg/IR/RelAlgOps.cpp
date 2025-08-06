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
    printer << " ";
    printer.printAttributeWithoutType(getTableOidAttr());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"table_name", "table_oid"});
    printer << " ";
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

ParseResult BaseTableOp::parse(OpAsmParser& parser, OperationState& result) {
    StringAttr tableName;
    IntegerAttr tableOid;
    Type resultType;
    
    if (parser.parseAttribute(tableName) ||
        parser.parseAttribute(tableOid) ||
        parser.parseArrow() ||
        parser.parseType(resultType)) {
        return failure();
    }
    
    result.addAttribute("table_name", tableName);
    result.addAttribute("table_oid", tableOid);
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
// MaterializeOp
//===----------------------------------------------------------------------===//

void MaterializeOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getRel());
    printer << " ";
    printer.printAttributeWithoutType(getColumnsAttr());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"columns"});
}

ParseResult MaterializeOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand rel;
    ArrayAttr columnsAttr;
    Type relType, resultType;
    
    if (parser.parseOperand(rel) ||
        parser.parseAttribute(columnsAttr) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(relType)) {
        return failure();
    }
    
    result.addAttribute("columns", columnsAttr);
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