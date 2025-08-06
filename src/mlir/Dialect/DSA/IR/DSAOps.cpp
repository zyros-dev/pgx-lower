#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "execution/logging.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace pgx::mlir::dsa;

//===----------------------------------------------------------------------===//
// ScanSourceOp
//===----------------------------------------------------------------------===//

void ScanSourceOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printAttributeWithoutType(getTableDescriptionAttr());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"table_description"});
}

ParseResult ScanSourceOp::parse(OpAsmParser& parser, OperationState& result) {
    StringAttr tableDescriptionAttr;
    Type resultType;
    
    if (parser.parseAttribute(tableDescriptionAttr) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }
    
    result.addAttribute("table_description", tableDescriptionAttr);
    result.addTypes(resultType);
    
    return success();
}

//===----------------------------------------------------------------------===//
// AtOp
//===----------------------------------------------------------------------===//

void AtOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getRecord());
    printer << "[";
    printer.printAttributeWithoutType(getColumnNameAttr());
    printer << "] -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs(), {"column_name"});
}

ParseResult AtOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand record;
    StringAttr columnName;
    Type recordType, resultType;
    
    if (parser.parseOperand(record) ||
        parser.parseLSquare() ||
        parser.parseAttribute(columnName) ||
        parser.parseRSquare() ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(recordType)) {
        return failure();
    }
    
    result.addAttribute("column_name", columnName);
    result.addTypes(resultType);
    
    if (parser.resolveOperand(record, recordType, result.operands)) {
        return failure();
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::print(OpAsmPrinter& printer) {
    printer << " (%record in ";
    printer.printOperand(getIterable());
    printer << ") : ";
    printer.printType(getIterable().getType());
    printer << " ";
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ForOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::Argument recordArg;
    OpAsmParser::UnresolvedOperand iterable;
    Type iterableType;
    StringRef recordArgName;
    
    if (parser.parseLParen() ||
        parser.parseKeyword(&recordArgName) ||
        parser.parseKeyword("in") ||
        parser.parseOperand(iterable) ||
        parser.parseRParen() ||
        parser.parseColon() ||
        parser.parseType(iterableType)) {
        return failure();
    }
    
    if (parser.resolveOperand(iterable, iterableType, result.operands)) {
        return failure();
    }
    
    Region* body = result.addRegion();
    recordArg.ssaName.name = recordArgName;
    recordArg.type = RecordType::get(parser.getContext());
    
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseRegion(*body, {recordArg}, /*enableNameShadowing=*/false)) {
        return failure();
    }
    
    ForOp::ensureTerminator(*body, parser.getBuilder(), result.location);
    return success();
}

//===----------------------------------------------------------------------===//
// CreateDSOp
//===----------------------------------------------------------------------===//

void CreateDSOp::print(OpAsmPrinter& printer) {
    printer << " -> ";
    printer.printType(getResult().getType());
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult CreateDSOp::parse(OpAsmParser& parser, OperationState& result) {
    Type resultType;
    
    if (parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }
    
    result.addTypes(resultType);
    return success();
}

//===----------------------------------------------------------------------===//
// DSAppendOp
//===----------------------------------------------------------------------===//

void DSAppendOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getBuilder());
    if (!getValues().empty()) {
        printer << ", ";
        printer.printOperands(getValues());
    }
    printer << " : ";
    printer.printType(getBuilder().getType());
    if (!getValues().empty()) {
        printer << ", ";
        interleaveComma(getValues().getTypes(), printer,
            [&](Type type) { printer.printType(type); });
    }
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult DSAppendOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand builder;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> values;
    Type builderType;
    SmallVector<Type, 4> valueTypes;
    
    if (parser.parseOperand(builder)) {
        return failure();
    }
    
    if (parser.parseOptionalComma()) {
        if (parser.parseOperandList(values)) {
            return failure();
        }
    }
    
    if (parser.parseColon() ||
        parser.parseType(builderType)) {
        return failure();
    }
    
    if (!values.empty()) {
        if (parser.parseComma() ||
            parser.parseTypeList(valueTypes)) {
            return failure();
        }
    }
    
    if (parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }
    
    if (parser.resolveOperand(builder, builderType, result.operands) ||
        parser.resolveOperands(values, valueTypes, parser.getCurrentLocation(), result.operands)) {
        return failure();
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// NextRowOp
//===----------------------------------------------------------------------===//

void NextRowOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getBuilder());
    printer << " : ";
    printer.printType(getBuilder().getType());
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult NextRowOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand builder;
    Type builderType;
    
    if (parser.parseOperand(builder) ||
        parser.parseColon() ||
        parser.parseType(builderType) ||
        parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }
    
    if (parser.resolveOperand(builder, builderType, result.operands)) {
        return failure();
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// FinalizeOp
//===----------------------------------------------------------------------===//

void FinalizeOp::print(OpAsmPrinter& printer) {
    printer << " ";
    printer.printOperand(getBuilder());
    printer << " -> ";
    printer.printType(getResult().getType());
    printer << " : ";
    printer.printType(getBuilder().getType());
    printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult FinalizeOp::parse(OpAsmParser& parser, OperationState& result) {
    OpAsmParser::UnresolvedOperand builder;
    Type builderType, resultType;
    
    if (parser.parseOperand(builder) ||
        parser.parseArrow() ||
        parser.parseType(resultType) ||
        parser.parseColon() ||
        parser.parseType(builderType) ||
        parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
    }
    
    result.addTypes(resultType);
    
    if (parser.resolveOperand(builder, builderType, result.operands)) {
        return failure();
    }
    
    return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOps.cpp.inc"