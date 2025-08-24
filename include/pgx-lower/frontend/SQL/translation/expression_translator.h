#pragma once

#include "translation_context.h"

// Forward declaration
extern "C" {
struct Expr;
}

namespace mlir {
class Value;
}

namespace pgx_lower::ast::expression {

// Translate a PostgreSQL expression to MLIR value
[[nodiscard]] auto translate(Expr* expr, TranslationContext& ctx) -> ::mlir::Value;

} // namespace pgx_lower::ast::expression