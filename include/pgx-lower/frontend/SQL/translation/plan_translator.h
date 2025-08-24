#pragma once

#include "translation_context.h"

// Forward declaration
extern "C" {
struct Plan;
}

namespace mlir {
class Operation;
}

namespace pgx_lower::ast::plan {

// Translate a PostgreSQL plan node to MLIR operation
[[nodiscard]] auto translate(Plan* plan, TranslationContext& ctx) -> ::mlir::Operation*;

} // namespace pgx_lower::ast::plan