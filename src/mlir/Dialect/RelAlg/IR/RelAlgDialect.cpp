#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "execution/logging.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx::mlir::relalg;

struct RelalgInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                                IRMapping& valueMapping) const override {
      return true;
   }
};

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"

void RelAlgDialect::initialize() {
    PGX_DEBUG("Initializing RelAlg dialect");
    
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
    >();
    
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"
    >();
    
    addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"
    >();
    
    addInterfaces<RelalgInlinerInterface>();
    columnManager.setContext(getContext());
    
    PGX_DEBUG("RelAlg dialect initialization complete");
}

// Custom attribute parsers and printers
::mlir::Attribute pgx::mlir::relalg::TableMetaDataAttr::parse(::mlir::AsmParser& parser, ::mlir::Type type) {
    StringAttr tableNameAttr;
    ArrayAttr columnsAttr;
    
    if (parser.parseLess() || 
        parser.parseAttribute(tableNameAttr) ||
        parser.parseComma() ||
        parser.parseAttribute(columnsAttr) ||
        parser.parseGreater()) {
        return Attribute();
    }
    
    return pgx::mlir::relalg::TableMetaDataAttr::get(parser.getContext(), 
                                                     tableNameAttr.str(), 
                                                     columnsAttr);
}

void pgx::mlir::relalg::TableMetaDataAttr::print(::mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(StringAttr::get(getContext(), getTableName()));
    printer << ",";
    printer.printAttribute(getColumns());
    printer << ">";
}

void pgx::mlir::relalg::ColumnDefAttr::print(::mlir::AsmPrinter& printer) const {
    printer << "<" << getName() << "," << getType();
    if (auto fromExisting = getFromExisting()) {
        printer << "," << fromExisting;
    }
    printer << ">";
}

::mlir::Attribute pgx::mlir::relalg::ColumnDefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
    mlir::SymbolRefAttr sym;
    mlir::Type t;
    mlir::Attribute fromExisting;
    
    if (parser.parseLess() || 
        parser.parseAttribute(sym) || 
        parser.parseComma() || 
        parser.parseType(t)) {
        return Attribute();
    }
    
    if (parser.parseOptionalComma().succeeded()) {
        if (parser.parseAttribute(fromExisting)) {
            return Attribute();
        }
    }
    
    if (parser.parseGreater()) return Attribute();
    
    return pgx::mlir::relalg::ColumnDefAttr::get(parser.getContext(), sym, t, fromExisting);
}

void pgx::mlir::relalg::ColumnRefAttr::print(::mlir::AsmPrinter& printer) const {
    printer << "<" << getName() << ">";
}

::mlir::Attribute pgx::mlir::relalg::ColumnRefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
    mlir::SymbolRefAttr sym;
    if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseGreater()) {
        return Attribute();
    }
    return pgx::mlir::relalg::ColumnRefAttr::get(parser.getContext(), sym);
}

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsDialect.cpp.inc"