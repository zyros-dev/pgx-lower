//===----------------------------------------------------------------------===//
// Simple RelAlg Dialect Implementation (Manual)
//===----------------------------------------------------------------------===//

#include "dialects/relalg/RelAlgDialectSimple.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::relalg;

//===----------------------------------------------------------------------===//
// RelAlg Dialect
//===----------------------------------------------------------------------===//

RelAlgDialect::RelAlgDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<RelAlgDialect>()) {
    initialize();
}

void RelAlgDialect::initialize() {
    addOperations<BaseTableOp, MaterializeOp>();
    addTypes<TupleStreamType>();
}

//===----------------------------------------------------------------------===//
// BaseTableOp
//===----------------------------------------------------------------------===//

void BaseTableOp::build(OpBuilder &builder, OperationState &state,
                       StringAttr tableIdentifier, Type resultType) {
    state.addAttribute("table_identifier", tableIdentifier);
    state.addTypes(resultType);
}

LogicalResult BaseTableOp::verify() {
    if (!getAttr("table_identifier"))
        return emitOpError("missing table_identifier attribute");
    return success();
}

void BaseTableOp::print(OpAsmPrinter &p) {
    p << " {table_identifier = " << getTableIdentifier() << "} : " << getType();
}

ParseResult BaseTableOp::parse(OpAsmParser &parser, OperationState &result) {
    StringAttr tableIdAttr;
    Type resultType;
    
    if (parser.parseLBrace() ||
        parser.parseKeyword("table_identifier") ||
        parser.parseEqual() ||
        parser.parseAttribute(tableIdAttr, "table_identifier", result.attributes) ||
        parser.parseRBrace() ||
        parser.parseColon() ||
        parser.parseType(resultType))
        return failure();
    
    result.addTypes(resultType);
    return success();
}

StringAttr BaseTableOp::getTableIdentifier() {
    return getAttr("table_identifier").cast<StringAttr>();
}

//===----------------------------------------------------------------------===//
// MaterializeOp
//===----------------------------------------------------------------------===//

void MaterializeOp::build(OpBuilder &builder, OperationState &state, Value input) {
    state.addOperands(input);
}

LogicalResult MaterializeOp::verify() {
    return success();
}

void MaterializeOp::print(OpAsmPrinter &p) {
    p << " " << getOperand() << " : " << getOperand().getType();
}

ParseResult MaterializeOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand operand;
    Type type;
    
    if (parser.parseOperand(operand) ||
        parser.parseColon() ||
        parser.parseType(type) ||
        parser.resolveOperand(operand, type, result.operands))
        return failure();
    
    return success();
}