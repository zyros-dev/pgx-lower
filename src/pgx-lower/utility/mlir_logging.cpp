#include "pgx-lower/utility/logging.h"
// logging.h forward-declares mlir::ModuleOp; this .cpp needs the complete
// type for the function signature + verify() call, so pull in the real
// MLIR builtin ops header (which defines ModuleOp fully).
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace pgx_lower::log {

auto verify_and_print(const mlir::Value VAL) -> void {
#ifndef PGX_RELEASE_MODE
    PGX_IO(AST_TRANSLATE);
    if (auto* def_op = VAL.getDefiningOp()) {
        const auto VERIFY_RESULT = mlir::verify(def_op);
        if (mlir::failed(VERIFY_RESULT)) {
            PGX_ERROR("MLIR verification FAILED for value");
            throw std::runtime_error("MLIR verification FAILED for value");
        }
    } else {
        PGX_LOG(AST_TRANSLATE, TRACE, "val had no defining op");
    }

    PGX_LOG(AST_TRANSLATE, TRACE, "finished verification - now printing.");
    try {
        std::string value_str;
        llvm::raw_string_ostream stream(value_str);
        VAL.print(stream);
        stream.flush();
        if (value_str.empty()) {
            PGX_LOG(AST_TRANSLATE, TRACE, "<empty print output>");
        } else {
            PGX_LOG(AST_TRANSLATE, TRACE, "%s", value_str.c_str());
        }
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during value print: %s", e.what());
    } catch (...) {
        PGX_ERROR("Unknown exception during value print");
    }
#endif
}

auto print_type(const mlir::Type VAL) -> void {
    std::string value_str;
    llvm::raw_string_ostream stream(value_str);
    VAL.print(stream);
    stream.flush();
    PGX_LOG(AST_TRANSLATE, TRACE, "%s", value_str.c_str());
}

auto type_to_string(const mlir::Type TYPE) -> std::string {
    std::string type_str;
    llvm::raw_string_ostream stream(type_str);
    TYPE.print(stream);
    stream.flush();
    return type_str;
}

auto value_to_string(const mlir::Value VAL) -> std::string {
    std::string value_str;
    llvm::raw_string_ostream stream(value_str);
    VAL.print(stream);
    stream.flush();
    return value_str;
}

auto verify_module_or_throw(::mlir::ModuleOp module, const char* phase_name, const char* error_context) -> bool {
#ifndef PGX_RELEASE_MODE
    if (mlir::failed(mlir::verify(module))) {
        std::string const error_msg = std::string(phase_name) + ": " + error_context;
        PGX_ERROR("%s", error_msg.c_str());
        throw std::runtime_error(error_msg);
    }
#endif
    return true;
}

} // namespace log
 // namespace pgx_lower
