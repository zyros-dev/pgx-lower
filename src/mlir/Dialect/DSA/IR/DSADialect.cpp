#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "execution/logging.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

// Forward declare DSA types
namespace pgx { namespace mlir { namespace dsa {
    class TableBuilderType;
    class VectorType;
    class TableType;
    class FlagType;
}}}

using namespace mlir;
using namespace ::mlir::dsa;

struct DSAInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                                IRMapping& valueMapping) const override {
      return true;
   }
};

// Type definitions are included in DSATypes.cpp, not here

void DSADialect::initialize() {
    PGX_DEBUG("Initializing DSA dialect");
    
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DSA/IR/DSAOps.cpp.inc"
    >();
    
    registerTypes();
    
    addInterfaces<DSAInlinerInterface>();
    
    PGX_DEBUG("DSA dialect initialization complete");
}

//===----------------------------------------------------------------------===//
// DSA Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

::mlir::Type DSADialect::parseType(::mlir::DialectAsmParser &parser) const {
    PGX_DEBUG("DSADialect::parseType() called");
    
    llvm::StringRef mnemonic;
    if (parser.parseKeyword(&mnemonic)) {
        return {};
    }
    
    // Parse DSA types by their mnemonic
    if (mnemonic == "table_builder") {
        if (parser.parseLess()) return {};
        ::mlir::Type rowType;
        if (parser.parseType(rowType)) return {};
        if (parser.parseGreater()) return {};
        auto tupleType = rowType.dyn_cast<::mlir::TupleType>();
        if (!tupleType) {
            parser.emitError(parser.getNameLoc(), "table_builder requires tuple type");
            return {};
        }
        return mlir::dsa::TableBuilderType::get(parser.getContext(), tupleType);
    }
    if (mnemonic == "vector") {
        if (parser.parseLess()) return {};
        ::mlir::Type elementType;
        if (parser.parseType(elementType)) return {};
        if (parser.parseGreater()) return {};
        return mlir::dsa::VectorType::get(parser.getContext(), elementType);
    }
    // Handle simple types
    if (mnemonic == "table") return mlir::dsa::TableType::get(parser.getContext());
    if (mnemonic == "flag") return mlir::dsa::FlagType::get(parser.getContext());
    
    parser.emitError(parser.getNameLoc(), "unknown DSA type: ") << mnemonic;
    return {};
}

void DSADialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const {
    PGX_DEBUG("DSADialect::printType() called");
    
    // Print DSA types by their mnemonic
    if (auto tableBuilder = type.dyn_cast<mlir::dsa::TableBuilderType>()) {
        printer << "table_builder<" << tableBuilder.getRowType() << ">";
        return;
    }
    if (auto vector = type.dyn_cast<mlir::dsa::VectorType>()) {
        printer << "vector<" << vector.getElementType() << ">";
        return;
    }
    if (type.isa<mlir::dsa::TableType>()) {
        printer << "table";
        return;
    }
    if (type.isa<mlir::dsa::FlagType>()) {
        printer << "flag";
        return;
    }
    
    PGX_ERROR("Failed to print unknown DSA type");
    printer << "<unknown DSA type>";
}

#include "mlir/Dialect/DSA/IR/DSAOpsDialect.cpp.inc"