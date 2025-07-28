#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::relalg;

void RelAlgDialect::initialize() {
    // Add types from TableGen
    addTypes<
#define GET_TYPEDEF_LIST
#include "RelAlgTypes.h.inc"
    >();
    
    // Add operations from TableGen
    addOperations<
#define GET_OP_LIST
#include "RelAlgOps.cpp.inc"
    >();
    
    // Load dependent dialects
    getContext()->loadDialect<pgx_lower::compiler::dialect::db::DBDialect>();
    getContext()->loadDialect<mlir::arith::ArithDialect>();
    getContext()->loadDialect<pgx_lower::compiler::dialect::tuples::TupleStreamDialect>();
}

// Parse attribute implementation
mlir::Attribute RelAlgDialect::parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const {
    // For now, we don't have any custom attributes to parse
    parser.emitError(parser.getCurrentLocation(), "RelAlg dialect has no custom attributes");
    return {};
}

// Print attribute implementation  
void RelAlgDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &os) const {
    // For now, we don't have any custom attributes to print
    llvm_unreachable("RelAlg dialect has no custom attributes");
}

// Include generated dialect implementation
#include "RelAlgDialect.cpp.inc"

// Include generated enum definitions
#include "RelAlgEnums.cpp.inc"