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

// TableMetaDataAttr removed - not implemented yet

// ColumnDefAttr print removed - not implemented yet

// ColumnDefAttr parsing removed - not implemented yet

// Type print/parse methods - custom implementation to prevent segfaults
void pgx::mlir::relalg::TupleStreamType::print(::mlir::AsmPrinter &printer) const {
    // No parameters to print for simple types
}

mlir::Type pgx::mlir::relalg::TupleStreamType::parse(::mlir::AsmParser &parser) {
    return get(parser.getContext());
}

void pgx::mlir::relalg::TupleType::print(::mlir::AsmPrinter &printer) const {
    // No parameters to print for simple types
}

mlir::Type pgx::mlir::relalg::TupleType::parse(::mlir::AsmParser &parser) {
    return get(parser.getContext());
}

void pgx::mlir::relalg::TableType::print(::mlir::AsmPrinter &printer) const {
    // No parameters to print for simple types
}

mlir::Type pgx::mlir::relalg::TableType::parse(::mlir::AsmParser &parser) {
    return get(parser.getContext());
}

void pgx::mlir::relalg::ColumnRefAttr::print(::mlir::AsmPrinter& printer) const {
    printer << "<" << getName() << ">";
}

::mlir::Attribute pgx::mlir::relalg::ColumnRefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
    std::string name;
    if (parser.parseLess() || parser.parseString(&name) || parser.parseGreater()) {
        return ::mlir::Attribute();
    }
    return pgx::mlir::relalg::ColumnRefAttr::get(parser.getContext(), name);
}

#include "mlir/Dialect/RelAlg/IR/RelAlgOpsDialect.cpp.inc"