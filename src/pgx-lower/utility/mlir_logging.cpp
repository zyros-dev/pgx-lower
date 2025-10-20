#include "pgx-lower/utility/logging.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace pgx_lower { namespace log {

auto verify_and_print(const mlir::Value val) -> void {
    PGX_IO(AST_TRANSLATE);
    if (auto* defOp = val.getDefiningOp()) {
        const auto verifyResult = mlir::verify(defOp);
        if (mlir::failed(verifyResult)) {
            PGX_ERROR("MLIR verification FAILED for value");
            throw std::runtime_error("MLIR verification FAILED for value");
        }
    } else {
        PGX_LOG(AST_TRANSLATE, TRACE, "val had no defining op");
    }

    PGX_LOG(AST_TRANSLATE, TRACE, "finished verification - now printing.");
    try {
        std::string valueStr;
        llvm::raw_string_ostream stream(valueStr);
        val.print(stream);
        stream.flush();
        if (valueStr.empty()) {
            PGX_LOG(AST_TRANSLATE, TRACE, "<empty print output>");
        } else {
            PGX_LOG(AST_TRANSLATE, TRACE, "%s", valueStr.c_str());
        }
    } catch (const std::exception& e) {
        PGX_ERROR("Exception during value print: %s", e.what());
    } catch (...) {
        PGX_ERROR("Unknown exception during value print");
    }
}

auto print_type(const mlir::Type val) -> void {
    std::string valueStr;
    llvm::raw_string_ostream stream(valueStr);
    val.print(stream);
    stream.flush();
    PGX_LOG(AST_TRANSLATE, TRACE, "%s", valueStr.c_str());
}

auto type_to_string(const mlir::Type type) -> std::string {
    std::string typeStr;
    llvm::raw_string_ostream stream(typeStr);
    type.print(stream);
    stream.flush();
    return typeStr;
}

auto value_to_string(const mlir::Value val) -> std::string {
    std::string valueStr;
    llvm::raw_string_ostream stream(valueStr);
    val.print(stream);
    stream.flush();
    return valueStr;
}

auto verify_module_or_throw(::mlir::ModuleOp module, const char* phase_name, const char* error_context) -> bool {
#ifndef PGX_RELEASE_MODE
    if (mlir::failed(mlir::verify(module))) {
        std::string error_msg = std::string(phase_name) + ": " + error_context;
        PGX_ERROR("%s", error_msg.c_str());
        throw std::runtime_error(error_msg);
    }
#endif
    return true;
}

}} // namespace pgx_lower::log
