#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "execution/logging.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx::db;

struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                                IRMapping& valueMapping) const override {
      return true;
   }
};

void DBDialect::initialize() {
    PGX_DEBUG("Initializing DB dialect");
    
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
    >();
    
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
    >();
    
    addInterfaces<DBInlinerInterface>();
    
    PGX_DEBUG("DB dialect initialization complete");
}

//===----------------------------------------------------------------------===//
// DB Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

mlir::Type DBDialect::parseType(mlir::DialectAsmParser &parser) const {
    llvm::StringRef mnemonic;
    if (parser.parseKeyword(&mnemonic)) {
        return {};
    }
    
    // Parse our simple types by mnemonic
    if (mnemonic == "external_source") {
        return ExternalSourceType::get(parser.getContext());
    }
    if (mnemonic == "nullable_i32") {
        return NullableI32Type::get(parser.getContext());
    }
    if (mnemonic == "nullable_i64") {
        return NullableI64Type::get(parser.getContext());
    }
    if (mnemonic == "nullable_f64") {
        return NullableF64Type::get(parser.getContext());
    }
    if (mnemonic == "nullable_bool") {
        return NullableBoolType::get(parser.getContext());
    }
    if (mnemonic == "sql_null") {
        return SqlNullType::get(parser.getContext());
    }
    
    parser.emitError(parser.getNameLoc(), "unknown db type: ") << mnemonic;
    return {};
}

void DBDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
    // Print our types by their mnemonic
    if (auto ext = type.dyn_cast<ExternalSourceType>()) {
        printer << "external_source";
        return;
    }
    if (type.isa<NullableI32Type>()) {
        printer << "nullable_i32";
        return;
    }
    if (type.isa<NullableI64Type>()) {
        printer << "nullable_i64";
        return;
    }
    if (type.isa<NullableF64Type>()) {
        printer << "nullable_f64";
        return;
    }
    if (type.isa<NullableBoolType>()) {
        printer << "nullable_bool";
        return;
    }
    if (type.isa<SqlNullType>()) {
        printer << "sql_null";
        return;
    }
    
    printer << "<unknown db type>";
}

#include "mlir/Dialect/DB/IR/DBOpsDialect.cpp.inc"