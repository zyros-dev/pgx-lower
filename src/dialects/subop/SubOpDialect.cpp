//===- SubOpDialect.cpp - SubOperator dialect implementation ----*- C++ -*-===//
//
// SubOperator dialect implementation
//
//===----------------------------------------------------------------------===//

#include "dialects/subop/SubOpDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::subop;

#include "SubOpDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SubOperator dialect initialization
//===----------------------------------------------------------------------===//

void SubOpDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "SubOpOps.cpp.inc"
    >();
    
    addTypes<
#define GET_TYPEDEF_LIST
#include "SubOpTypes.cpp.inc"
    >();
}

// Default implementations for attribute parsing/printing
Attribute SubOpDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
    parser.emitError(parser.getNameLoc(), "unknown SubOp attribute");
    return Attribute();
}

void SubOpDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
    llvm_unreachable("unknown SubOp attribute");
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "SubOpTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "SubOpOps.cpp.inc"