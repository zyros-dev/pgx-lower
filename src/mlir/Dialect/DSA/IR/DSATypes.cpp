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
    PGX_INFO("[DSA] TableBuilderType::print() called");
    
    PGX_INFO("[DSA] Getting rowType from TableBuilderType");
    auto rowType = getRowType();
    
    if (!rowType) {
        PGX_ERROR("[DSA] TableBuilderType has null rowType!");
        printer << "<null_row_type>";
        return;
    }
    
    PGX_INFO("[DSA] TableBuilderType rowType is valid, printing");
    printer << "<";
    
    PGX_INFO("[DSA] About to call printer.printType() on rowType");
    printer.printType(rowType);
    PGX_INFO("[DSA] printer.printType() completed successfully");
    
    printer << ">";
    PGX_INFO("[DSA] TableBuilderType::print() completed successfully");
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
    PGX_INFO("[DSA] TableType::print() called");
    
    PGX_INFO("[DSA] Getting rowType from TableType");
    auto rowType = getRowType();
    
    if (!rowType) {
        PGX_ERROR("[DSA] TableType has null rowType!");
        printer << "<null_row_type>";
        return;
    }
    
    PGX_INFO("[DSA] TableType rowType is valid, printing");
    printer << "<";
    
    PGX_INFO("[DSA] About to call printer.printType() on rowType");
    printer.printType(rowType);
    PGX_INFO("[DSA] printer.printType() completed successfully");
    
    printer << ">";
    PGX_INFO("[DSA] TableType::print() completed successfully");
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