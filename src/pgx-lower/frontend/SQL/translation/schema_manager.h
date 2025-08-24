#pragma once

#include <string>
#include <vector>
#include "pgx-lower/frontend/SQL/translation/translation_context.h"

// Forward declarations
struct PlannedStmt;

namespace pgx_lower::frontend::sql {

// PostgreSQL schema access functions - CRITICAL: Preserve all catalog access patterns
auto getTableNameFromRTE(PlannedStmt* currentPlannedStmt, int varno) -> std::string;
auto getColumnNameFromSchema(PlannedStmt* currentPlannedStmt, int varno, int varattno) -> std::string;
auto getAllTableColumnsFromSchema(PlannedStmt* currentPlannedStmt, int scanrelid) 
    -> std::vector<ColumnInfo>;

} // namespace pgx_lower::frontend::sql