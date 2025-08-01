//===- DSADialect.cpp - Data Structure Access dialect impl -------*- C++ -*-===//
//
// Data Structure Access dialect implementation
//
//===----------------------------------------------------------------------===//

#include "dialects/dsa/DSADialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::dsa;

#include "DSADialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Data Structure Access dialect initialization
//===----------------------------------------------------------------------===//

void DSADialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "DSAOps.cpp.inc"
    >();
    
    addTypes<
#define GET_TYPEDEF_LIST
#include "DSATypes.cpp.inc"
    >();
}

// Default implementations for attribute parsing/printing
Attribute DSADialect::parseAttribute(DialectAsmParser &parser, Type type) const {
    parser.emitError(parser.getNameLoc(), "unknown DSA attribute");
    return Attribute();
}

void DSADialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
    llvm_unreachable("unknown DSA attribute");
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "DSATypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "DSAOps.cpp.inc"