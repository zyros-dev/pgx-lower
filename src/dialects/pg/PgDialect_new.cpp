#include "dialects/pg/PgDialect_new.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;
using namespace mlir::pg;

//===----------------------------------------------------------------------===//
// PostgreSQL Dialect
//===----------------------------------------------------------------------===//

void PgDialect::initialize() {
    // TableGen generated operation and type registration
    addOperations<
#define GET_OP_LIST
#include "PgOps.cpp.inc"
    >();
    
    addTypes<
#define GET_TYPEDEF_LIST
#include "PgTypes.cpp.inc"
    >();
}

PgDialect::PgDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<PgDialect>()) {
    initialize();
}

Type PgDialect::parseType(DialectAsmParser &parser) const {
    // Use TableGen-generated type parser
    StringRef mnemonic;
    Type genType;
    OptionalParseResult parseResult = generatedTypeParser(parser, &mnemonic, genType);
    if (parseResult.has_value())
        return genType;
    
    return Type();
}

void PgDialect::printType(Type type, DialectAsmPrinter &printer) const {
    // Use TableGen-generated type printer
    if (generatedTypePrinter(type, printer).succeeded())
        return;
    
    // Fallback for unknown types
    printer << "unknown_type";
}

// Include TableGen generated definitions
#define GET_OP_CLASSES
#include "PgOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PgTypes.cpp.inc"