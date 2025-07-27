//===----------------------------------------------------------------------===//
// Simple RelAlg Dialect Implementation (Manual)
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_RELALG_RELALGDIALECTSIMPLE_H
#define DIALECTS_RELALG_RELALGDIALECTSIMPLE_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace relalg {

//===----------------------------------------------------------------------===//
// RelAlg Dialect
//===----------------------------------------------------------------------===//

class RelAlgDialect : public Dialect {
public:
    explicit RelAlgDialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "relalg"; }

    void initialize();
};

//===----------------------------------------------------------------------===//
// RelAlg Types
//===----------------------------------------------------------------------===//

class TupleStreamType : public Type::TypeBase<TupleStreamType, Type, TypeStorage> {
public:
    using Base::Base;
    static constexpr StringLiteral name = "relalg.tuplestream";
};

//===----------------------------------------------------------------------===//
// RelAlg Operations
//===----------------------------------------------------------------------===//

class BaseTableOp : public Op<BaseTableOp> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "relalg.basetable"; }
    
    static void build(OpBuilder &builder, OperationState &state, 
                     StringAttr tableIdentifier, Type resultType);
    
    LogicalResult verify();
    void print(OpAsmPrinter &p);
    static ParseResult parse(OpAsmParser &parser, OperationState &result);
    
    StringAttr getTableIdentifier();
};

class MaterializeOp : public Op<MaterializeOp> {
public:
    using Op::Op;
    static StringRef getOperationName() { return "relalg.materialize"; }
    
    static void build(OpBuilder &builder, OperationState &state, Value input);
    
    LogicalResult verify();
    void print(OpAsmPrinter &p);
    static ParseResult parse(OpAsmParser &parser, OperationState &result);
};

} // namespace relalg
} // namespace mlir

#endif // DIALECTS_RELALG_RELALGDIALECTSIMPLE_H