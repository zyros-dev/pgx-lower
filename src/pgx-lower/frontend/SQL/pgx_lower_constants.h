#ifndef PGX_LOWER_CONSTANTS_H
#define PGX_LOWER_CONSTANTS_H

namespace pgx_lower::frontend::sql::constants {

// =============================================================================
// Column naming patterns
// =============================================================================

constexpr auto GENERATED_COLUMN_PREFIX = "col_";
constexpr auto COMPUTED_EXPRESSION_SCOPE = "map";
constexpr auto AGGREGATION_RESULT_COLUMN = "aggr_result";

// =============================================================================
// POSTGRESQL INTERNAL CONSTANTS
// =============================================================================

constexpr auto POSTGRESQL_VARNO_OFFSET = 1;
constexpr auto POSTGRESQL_VARHDRSZ = 4;
constexpr auto INVALID_TYPMOD = -1;

constexpr auto INVALID_VARNO = 0;
constexpr auto INVALID_VARATTNO = 0;
constexpr Oid PG_TEXT_ARRAY_OID = 1009;
constexpr Oid FIRST_NORMAL_OBJECT_ID = 16384;

constexpr auto BOOL_AND_EXPR = 0;
constexpr auto BOOL_OR_EXPR = 1;
constexpr auto BOOL_NOT_EXPR = 2;

// PostgreSQL Null Test Constants
constexpr auto PG_IS_NOT_NULL = 1;

// =============================================================================
// POSTGRESQL TYPE SYSTEM CONSTANTS
// =============================================================================

constexpr auto MIN_NUMERIC_PRECISION = 1;
constexpr auto NUMERIC_PRECISION_SHIFT = 16;
constexpr auto NUMERIC_PRECISION_MASK = 0xFFFF;
constexpr auto NUMERIC_SCALE_MASK = 0xFFFF;

constexpr auto AVERAGE_DAYS_PER_MONTH = 29.53;

constexpr auto BOOL_BIT_WIDTH = 1;
constexpr auto INT2_BIT_WIDTH = 16;
constexpr auto INT4_BIT_WIDTH = 32;
constexpr auto INT8_BIT_WIDTH = 64;

constexpr auto BOOL_TRUE_VALUE = 1;
constexpr auto BOOL_FALSE_VALUE = 0;

constexpr auto TIMESTAMP_PRECISION_SECOND = 0;
constexpr auto TIMESTAMP_PRECISION_MILLI_MIN = 1;
constexpr auto TIMESTAMP_PRECISION_MILLI_MAX = 3;
constexpr auto TIMESTAMP_PRECISION_MICRO_MIN = 4;
constexpr auto TIMESTAMP_PRECISION_MICRO_MAX = 6;
constexpr auto TIMESTAMP_PRECISION_NANO_MIN = 7;
constexpr auto TIMESTAMP_PRECISION_NANO_MAX = 9;

// =============================================================================
// STRING CONSTANTS
// =============================================================================

// Aggregate Function OIDs
constexpr auto PG_COUNT_ANY_OID = 2147; // COUNT(*)
constexpr auto PG_COUNT_OID = 2803; // COUNT(expr)

constexpr auto UNIT_TEST_TABLE_PREFIX = "test_table_";
constexpr auto FALLBACK_TABLE_PREFIX = "table_";

// Generated Names
constexpr auto EXPRESSION_COLUMN_PREFIX = "expr_";
constexpr auto TABLE_OID_SEPARATOR = "|oid:";
constexpr auto QUERY_FUNCTION_NAME = "main";

// Unit Test Column Configuration
constexpr auto UNIT_TEST_COLUMN_NOT_NULL = false;
constexpr auto PG_ATTNAME_NOT_MISSING_OK = false;

// =============================================================================
// CONFIGURATION LIMITS
// =============================================================================

constexpr auto MAX_QUERY_COLUMNS = 1000;
constexpr auto MAX_COLUMN_INDEX = 1000;
constexpr auto MAX_LIST_LENGTH = 1000;
constexpr auto MAX_LIMIT_COUNT = 1000000;
constexpr auto DEFAULT_LIMIT_COUNT = 10;

constexpr auto MAX_NUMERIC_PRECISION = 21;
constexpr auto MAX_NUMERIC_UNCONSTRAINED_SCALE = 16;
} // namespace pgx_lower::frontend::sql::constants

#endif // PGX_LOWER_CONSTANTS_H