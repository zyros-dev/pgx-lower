#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "pgx-lower/utility/logging.h"

using namespace mlir;
using namespace ::mlir::relalg;

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
struct ArithCmpICmpInterface
   : public CmpOpInterface::ExternalModel<ArithCmpICmpInterface, mlir::arith::CmpIOp> {
   bool isEqualityPred(::mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      return cmpOp.getPredicate() == mlir::arith::CmpIPredicate::eq;
   }
   bool isLessPred(::mlir::Operation* op, bool eq) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      switch (cmpOp.getPredicate()) {
         case mlir::arith::CmpIPredicate::sle:
         case mlir::arith::CmpIPredicate::ule:
            return eq;
         case mlir::arith::CmpIPredicate::ult:
         case mlir::arith::CmpIPredicate::slt:
            return !eq;
         default: return false;
      }
   }
   bool isGreaterPred(::mlir::Operation* op, bool eq) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      switch (cmpOp.getPredicate()) {
         case mlir::arith::CmpIPredicate::sge:
         case mlir::arith::CmpIPredicate::uge:
            return eq;
         case mlir::arith::CmpIPredicate::ugt:
         case mlir::arith::CmpIPredicate::sgt:
            return !eq;
         default: return false;
      }
   }
   ::mlir::Value getLeft(::mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      return cmpOp.getLhs();
   }
   ::mlir::Value getRight(::mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpIOp>(op);
      return cmpOp.getRhs();
   }
};
struct ArithCmpFCmpInterface
   : public CmpOpInterface::ExternalModel<ArithCmpFCmpInterface, mlir::arith::CmpFOp> {
   bool isEqualityPred(::mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      return cmpOp.getPredicate() == mlir::arith::CmpFPredicate::OEQ || cmpOp.getPredicate() == mlir::arith::CmpFPredicate::UEQ;
   }
   bool isLessPred(::mlir::Operation* op, bool eq) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      switch (cmpOp.getPredicate()) {
         case mlir::arith::CmpFPredicate::ULE:
         case mlir::arith::CmpFPredicate::OLE:
            return eq;
         case mlir::arith::CmpFPredicate::ULT:
         case mlir::arith::CmpFPredicate::OLT:
            return !eq;
         default: return false;
      }
   }
   bool isGreaterPred(::mlir::Operation* op, bool eq) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      switch (cmpOp.getPredicate()) {
         case mlir::arith::CmpFPredicate::UGE:
         case mlir::arith::CmpFPredicate::OGE:
            return eq;
         case mlir::arith::CmpFPredicate::UGT:
         case mlir::arith::CmpFPredicate::OGT:
            return !eq;
         default: return false;
      }
   }
   ::mlir::Value getLeft(::mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      return cmpOp.getLhs();
   }
   ::mlir::Value getRight(::mlir::Operation* op) const {
      auto cmpOp = mlir::cast<mlir::arith::CmpFOp>(op);
      return cmpOp.getRhs();
   }
};
#define GET_ATTRDEF_CLASSES
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"
void RelAlgDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
      >();
   registerTypes();
   addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"
      >();
   addInterfaces<RelalgInlinerInterface>();
   columnManager.setContext(getContext());
   getContext()->loadDialect<mlir::db::DBDialect>();
   getContext()->loadDialect<mlir::dsa::DSADialect>();
   getContext()->loadDialect<mlir::arith::ArithDialect>();
   mlir::arith::CmpFOp::attachInterface<ArithCmpFCmpInterface>(*getContext());
   mlir::arith::CmpIOp::attachInterface<ArithCmpICmpInterface>(*getContext());
}

::mlir::Attribute mlir::relalg::TableMetaDataAttr::parse(::mlir::AsmParser& parser, ::mlir::Type type) {
   StringAttr attr;
   if (parser.parseLess() || parser.parseAttribute(attr) || parser.parseGreater()) return Attribute();
   return mlir::relalg::TableMetaDataAttr::get(parser.getContext(), runtime::TableMetaData::deserialize(attr.str()));
}
void mlir::relalg::TableMetaDataAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<";
   if (getMeta()) {
      printer.printAttribute(StringAttr::get(getContext(), getMeta()->serialize()));
   } else {
      printer.printAttribute(StringAttr::get(getContext(), "{}"));
   }
   printer << ">";
}
void mlir::relalg::ColumnDefAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getName();
   if (getColumnPtr()) {
      printer << "," << getColumn().type;
   } else {
      printer << ",<null>";
   }
   if (auto fromexisting = getFromExisting()) {
      printer << "," << fromexisting;
   }
   printer << ">";
}
::mlir::Attribute mlir::relalg::ColumnDefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   ::mlir::Type t;
   ::mlir::ArrayAttr fromExisting;
   if (parser.parseLess() || parser.parseAttribute(sym)||parser.parseComma()||parser.parseType(t)) return Attribute();
   if (parser.parseOptionalComma().succeeded()) {
      if (parser.parseAttribute(fromExisting)) {
         return Attribute();
      }
   }
   if (parser.parseGreater()) return Attribute();
   auto columnDef= parser.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager().createDef(sym, fromExisting);
   columnDef.getColumn().type=t;
   return columnDef;
}
void mlir::relalg::ColumnRefAttr::print(::mlir::AsmPrinter& printer) const {
   // Check if name is valid before printing
   if (getName()) {
      printer << "<" << getName() << ">";
   } else {
      printer << "<null>";
   }
}
::mlir::Attribute mlir::relalg::ColumnRefAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseGreater()) return Attribute();
   return parser.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager().createRef(sym);
}
void mlir::relalg::SortSpecificationAttr::print(::mlir::AsmPrinter& printer) const {
   printer << "<" << getAttr().getName() << "," << stringifyEnum(getSortSpec()) << ">";
}

// SortSpecPtrAttr print/parse removed - using UI64Attr instead

::mlir::Attribute mlir::relalg::SortSpecificationAttr::parse(::mlir::AsmParser& parser, ::mlir::Type odsType) {
   mlir::SymbolRefAttr sym;
   std::string sortSpecDescr;
   if (parser.parseLess() || parser.parseAttribute(sym) || parser.parseComma() || parser.parseKeywordOrString(&sortSpecDescr) || parser.parseGreater()) {
      return ::mlir::Attribute();
   }
   auto sortSpec = symbolizeSortSpec(sortSpecDescr);
   if (!sortSpec.has_value()) {
      return {};
   }
   auto columnRefAttr = parser.getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager().createRef(sym);
   return mlir::relalg::SortSpecificationAttr::get(parser.getContext(), columnRefAttr, sortSpec.value());
}
void RelAlgDialect::printAttribute(Attribute attr, DialectAsmPrinter& printer) const {
   if (auto columnDefAttr = attr.dyn_cast<mlir::relalg::ColumnDefAttr>()) {
      columnDefAttr.print(printer);
   } else if (auto columnRefAttr = attr.dyn_cast<mlir::relalg::ColumnRefAttr>()) {
      columnRefAttr.print(printer);
   } else if (auto sortSpecAttr = attr.dyn_cast<mlir::relalg::SortSpecificationAttr>()) {
      sortSpecAttr.print(printer);
   } else if (auto tableMetaAttr = attr.dyn_cast<mlir::relalg::TableMetaDataAttr>()) {
      tableMetaAttr.print(printer);
   } else {
      PGX_ERROR("Unknown relalg attribute type - cannot print");
      llvm_unreachable("unknown relalg attribute");
   }
}

Attribute RelAlgDialect::parseAttribute(DialectAsmParser& parser, Type type) const {
   llvm::StringRef attrTag;
   if (parser.parseKeyword(&attrTag))
      return Attribute();

   if (attrTag == "column_def")
      return mlir::relalg::ColumnDefAttr::parse(parser, type);
   if (attrTag == "column_ref")
      return mlir::relalg::ColumnRefAttr::parse(parser, type);
   if (attrTag == "sort_spec")
      return mlir::relalg::SortSpecificationAttr::parse(parser, type);
   if (attrTag == "table_meta")
      return mlir::relalg::TableMetaDataAttr::parse(parser, type);

   parser.emitError(parser.getNameLoc(), "unknown relalg attribute: ") << attrTag;
   return Attribute();
}

#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsDialect.cpp.inc"
