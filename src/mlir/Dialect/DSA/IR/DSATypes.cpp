#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "execution/logging.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx::mlir::dsa;

//===----------------------------------------------------------------------===//
// TableBuilderType Custom Assembly Format
//===----------------------------------------------------------------------===//

Type TableBuilderType::parse(AsmParser &parser) {
    // Parse format: !dsa.table_builder<TupleType>
    if (parser.parseLess())
        return {};
    
    TupleType rowType;
    if (parser.parseType(rowType))
        return {};
    
    if (parser.parseGreater())
        return {};
    
    return TableBuilderType::get(parser.getContext(), rowType);
}

void TableBuilderType::print(AsmPrinter &printer) const {
    // Simple safe printing to avoid infinite recursion
    printer << "<table_builder>";
}

//===----------------------------------------------------------------------===//
// TableType Custom Assembly Format
//===----------------------------------------------------------------------===//

Type TableType::parse(AsmParser &parser) {
    // Parse format: !dsa.table<TupleType>
    if (parser.parseLess())
        return {};
    
    TupleType rowType;
    if (parser.parseType(rowType))
        return {};
    
    if (parser.parseGreater())
        return {};
    
    return TableType::get(parser.getContext(), rowType);
}

void TableType::print(AsmPrinter &printer) const {
    // Simple safe printing to avoid infinite recursion
    printer << "<table>";
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"

namespace pgx {
namespace mlir {
namespace dsa {
void DSADialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
    >();
}
} // namespace dsa
} // namespace mlir
} // namespace pgx