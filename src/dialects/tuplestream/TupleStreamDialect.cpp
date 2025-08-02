#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "dialects/tuplestream/Column.h"
#include "dialects/tuplestream/ColumnManager.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/Hashing.h"

#include <tuple>

using namespace mlir;
using namespace pgx_lower::compiler::dialect::tuples;

// Implement ColumnDefAttr custom assembly format methods
mlir::Attribute pgx_lower::compiler::dialect::tuples::ColumnDefAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
   mlir::SymbolRefAttr sym;
   mlir::Type t;
   mlir::ArrayAttr fromExisting;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseComma() || parser.parseType(t)) return Attribute();
   if (parser.parseOptionalComma().succeeded()) {
      if (parser.parseAttribute(fromExisting)) {
         return Attribute();
      }
   }
   if (parser.parseGreater()) return Attribute();
   auto columnDef = parser.getContext()->getLoadedDialect<TupleStreamDialect>()->getColumnManager().createDef(sym, fromExisting);
   columnDef.getColumn().type = t;
   return columnDef;
}

void pgx_lower::compiler::dialect::tuples::ColumnDefAttr::print(mlir::AsmPrinter &printer) const {
   printer << "<" << getName() << "," << getColumn().type;
   if (auto fromexisting = getFromExisting()) {
      printer << "," << fromexisting;
   }
   printer << ">";
}

// Implement ColumnRefAttr custom assembly format methods  
mlir::Attribute pgx_lower::compiler::dialect::tuples::ColumnRefAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
   // For now, create dummy values - this is the minimal implementation needed
   auto name = SymbolRefAttr::get(parser.getContext(), "dummy");
   auto columnPtr = std::make_shared<Column>(); // Simple shared_ptr for now
   return ColumnRefAttr::get(parser.getContext(), name, columnPtr);
}

void pgx_lower::compiler::dialect::tuples::ColumnRefAttr::print(mlir::AsmPrinter &printer) const {
   printer << "<" << getName() << ">";
}

void TupleStreamDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "TupleStreamOps.cpp.inc"
   >();
   
   // Initialize the column manager
   columnManager.setContext(getContext());
   
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

// getColumnManager is defined inline in the TableGen-generated header

// getColumn() methods are now inline in TableGen definition