#pragma once

#include "mlir/IR/Types.h"
#include <cstdint>

namespace lingodb::utility {

[[nodiscard]] uint32_t mlir_type_to_pg_oid(mlir::Type type);

struct SortOperatorSpec {
    uint32_t comparison_op;
    uint32_t collation;
};

[[nodiscard]] SortOperatorSpec get_sort_operator(uint32_t type_oid, bool ascending);

} // namespace lingodb::utility
