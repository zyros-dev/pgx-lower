#include "dialects/tuplestream/Column.h"
#include "dialects/tuplestream/TupleStreamDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/Hashing.h"

#include <tuple>

using namespace mlir;
using namespace pgx_lower::compiler::dialect::tuples;

// Implement ColumnDefAttr custom assembly format methods
mlir::Attribute pgx_lower::compiler::dialect::tuples::ColumnDefAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
   // For now, create dummy values - this is the minimal implementation needed
   auto name = SymbolRefAttr::get(parser.getContext(), "dummy");
   auto columnType = parser.getBuilder().getI32Type(); // Simple type for now
   auto fromExisting = UnitAttr::get(parser.getContext());
   return ColumnDefAttr::get(parser.getContext(), name, columnType, fromExisting);
}

void pgx_lower::compiler::dialect::tuples::ColumnDefAttr::print(mlir::AsmPrinter &printer) const {
   printer << "@column";
}

// Implement ColumnRefAttr custom assembly format methods  
mlir::Attribute pgx_lower::compiler::dialect::tuples::ColumnRefAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
   // For now, create dummy values - this is the minimal implementation needed
   auto name = SymbolRefAttr::get(parser.getContext(), "dummy");
   auto columnType = parser.getBuilder().getI32Type(); // Simple type for now
   return ColumnRefAttr::get(parser.getContext(), name, columnType);
}

void pgx_lower::compiler::dialect::tuples::ColumnRefAttr::print(mlir::AsmPrinter &printer) const {
   printer << "@ref";
}

void TupleStreamDialect::initialize() {
   // addOperations<
   //    // Add operations when needed
   // >();
   
   // TODO: Load ColumnManager if needed
   registerTypes();
   registerAttrs();
}

void TupleStreamDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "TupleStreamTypes.cpp.inc"
   >();
}

void TupleStreamDialect::registerAttrs() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "TupleStreamAttrs.cpp.inc"
   >();
}

#include "TupleStreamDialect.cpp.inc"

// Include type storage implementations
#define GET_TYPEDEF_CLASSES
#include "TupleStreamTypes.cpp.inc"

// Include attribute implementations  
#define GET_ATTRDEF_CLASSES
#include "TupleStreamAttrs.cpp.inc"