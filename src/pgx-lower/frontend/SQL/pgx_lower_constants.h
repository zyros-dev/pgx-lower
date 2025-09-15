#ifndef PGX_LOWER_CONSTANTS_H
#define PGX_LOWER_CONSTANTS_H

// PostgreSQL catalog headers should already be included by the main translator file
// TODO: When we have this many constants, maybe we should create like PlusOid<data_type>()
// rather than a gazillion constants

namespace pgx_lower::frontend::sql::constants {

// =============================================================================
// ARITHMETIC OPERATOR OIDs
// =============================================================================

// Integer Addition Operators
constexpr Oid PG_INT4_PLUS_OID = 551;
constexpr Oid PG_INT8_PLUS_OID = 684;

// Integer Subtraction Operators
constexpr Oid PG_INT4_MINUS_OID = 552;
constexpr Oid PG_INT4_MINUS_ALT_OID = 555;
constexpr Oid PG_INT8_MINUS_OID = 688;

// Integer Multiplication Operators
constexpr Oid PG_INT4_MUL_OID = 514;
constexpr Oid PG_INT8_MUL_OID = 686;

// Integer Division Operators
constexpr Oid PG_INT4_DIV_OID = 527;
constexpr Oid PG_INT4_DIV_ALT_OID = 528;
constexpr Oid PG_INT8_DIV_OID = 689;

// Integer Modulo Operators
constexpr Oid PG_INT4_MOD_OID = 529;
constexpr Oid PG_INT4_MOD_ALT_OID = 530;
constexpr Oid PG_INT8_MOD_OID = 690;

// =============================================================================
// COMPARISON OPERATOR OIDs
// =============================================================================

// Integer Equality Operators
constexpr Oid PG_INT2_EQ_OID = 92;
constexpr Oid PG_INT4_EQ_OID = 96;
constexpr Oid PG_INT8_EQ_OID = 410;
constexpr Oid PG_TEXT_EQ_OID = 98;

// Integer Inequality Operators
constexpr Oid PG_INT2_NE_OID = 519;
constexpr Oid PG_INT4_NE_OID = 518;
constexpr Oid PG_INT8_NE_OID = 411;
constexpr Oid PG_TEXT_NE_OID = 531;

// Integer Less Than Operators
constexpr Oid PG_INT2_LT_OID = 95;
constexpr Oid PG_INT4_LT_OID = 97;
constexpr Oid PG_INT8_LT_OID = 412;
constexpr Oid PG_TEXT_LT_OID = 664;

// Integer Greater Than Operators
constexpr Oid PG_INT2_GT_OID = 520;
constexpr Oid PG_INT4_GT_OID = 521;
constexpr Oid PG_INT8_GT_OID = 413;
constexpr Oid PG_TEXT_GT_OID = 666;

// Integer Less Than or Equal Operators
constexpr Oid PG_INT2_LE_OID = 522;
constexpr Oid PG_INT4_LE_OID = 523;
constexpr Oid PG_INT8_LE_OID = 414;
constexpr Oid PG_TEXT_LE_OID = 665;

// Integer Greater Than or Equal Operators
constexpr Oid PG_INT2_GE_OID = 524;
constexpr Oid PG_INT4_GE_OID = 525;
constexpr Oid PG_INT8_GE_OID = 415;
constexpr Oid PG_TEXT_GE_OID = 667;

// Mixed Type Comparison Operators (INT4/INT8)
constexpr Oid PG_INT4_INT8_EQ_OID = 15;    // int48eq
constexpr Oid PG_INT4_INT8_NE_OID = 36;    // int48ne
constexpr Oid PG_INT4_INT8_LT_OID = 37;    // int48lt
constexpr Oid PG_INT4_INT8_GT_OID = 76;    // int48gt
constexpr Oid PG_INT4_INT8_LE_OID = 80;    // int48le
constexpr Oid PG_INT4_INT8_GE_OID = 82;    // int48ge

constexpr Oid PG_INT8_INT4_EQ_OID = 416;   // int84eq
constexpr Oid PG_INT8_INT4_NE_OID = 417;   // int84ne
constexpr Oid PG_INT8_INT4_LT_OID = 418;   // int84lt
constexpr Oid PG_INT8_INT4_GT_OID = 419;   // int84gt
constexpr Oid PG_INT8_INT4_LE_OID = 420;   // int84le
constexpr Oid PG_INT8_INT4_GE_OID = 430;   // int84ge

// Mixed Type Comparison Operators (INT2/INT4)
constexpr Oid PG_INT2_INT4_EQ_OID = 532;   // int24eq
constexpr Oid PG_INT4_INT2_EQ_OID = 533;   // int42eq
constexpr Oid PG_INT2_INT4_LT_OID = 534;   // int24lt
constexpr Oid PG_INT4_INT2_LT_OID = 535;   // int42lt
constexpr Oid PG_INT2_INT4_GT_OID = 536;   // int24gt
constexpr Oid PG_INT4_INT2_GT_OID = 537;   // int42gt
constexpr Oid PG_INT2_INT4_NE_OID = 538;   // int24ne
constexpr Oid PG_INT4_INT2_NE_OID = 539;   // int42ne
constexpr Oid PG_INT2_INT4_LE_OID = 540;   // int24le
constexpr Oid PG_INT4_INT2_LE_OID = 541;   // int42le
constexpr Oid PG_INT2_INT4_GE_OID = 542;   // int24ge
constexpr Oid PG_INT4_INT2_GE_OID = 543;   // int42ge

// Mixed Type Comparison Operators (INT2/INT8)
constexpr Oid PG_INT2_INT8_EQ_OID = 1862;  // int28eq
constexpr Oid PG_INT2_INT8_NE_OID = 1863;  // int28ne
constexpr Oid PG_INT2_INT8_LT_OID = 1864;  // int28lt
constexpr Oid PG_INT2_INT8_GT_OID = 1865;  // int28gt
constexpr Oid PG_INT2_INT8_LE_OID = 1866;  // int28le
constexpr Oid PG_INT2_INT8_GE_OID = 1867;  // int28ge

constexpr Oid PG_INT8_INT2_EQ_OID = 1868;  // int82eq
constexpr Oid PG_INT8_INT2_NE_OID = 1869;  // int82ne
constexpr Oid PG_INT8_INT2_LT_OID = 1870;  // int82lt
constexpr Oid PG_INT8_INT2_GT_OID = 1871;  // int82gt
constexpr Oid PG_INT8_INT2_LE_OID = 1872;  // int82le
constexpr Oid PG_INT8_INT2_GE_OID = 1873;  // int82ge

// Float Equality Operators
constexpr Oid PG_FLOAT4_EQ_OID = 620;
constexpr Oid PG_FLOAT8_EQ_OID = 670;

// Float Inequality Operators
constexpr Oid PG_FLOAT4_NE_OID = 621;
constexpr Oid PG_FLOAT8_NE_OID = 671;

// Float Less Than Operators
constexpr Oid PG_FLOAT4_LT_OID = 622;
constexpr Oid PG_FLOAT8_LT_OID = 672;

// Float Greater Than Operators
constexpr Oid PG_FLOAT4_GT_OID = 623;
constexpr Oid PG_FLOAT8_GT_OID = 674;

// Float Less Than or Equal Operators
constexpr Oid PG_FLOAT4_LE_OID = 624;
constexpr Oid PG_FLOAT8_LE_OID = 673;

// Float Greater Than or Equal Operators
constexpr Oid PG_FLOAT4_GE_OID = 625;
constexpr Oid PG_FLOAT8_GE_OID = 675;

// Numeric/Decimal Equality Operators
constexpr Oid PG_NUMERIC_EQ_OID = 1752;

// Numeric/Decimal Inequality Operators
constexpr Oid PG_NUMERIC_NE_OID = 1753;

// Numeric/Decimal Less Than Operators
constexpr Oid PG_NUMERIC_LT_OID = 1754;

// Numeric/Decimal Greater Than Operators
constexpr Oid PG_NUMERIC_GT_OID = 1756;

// Numeric/Decimal Less Than or Equal Operators
constexpr Oid PG_NUMERIC_LE_OID = 1755;

// Numeric/Decimal Greater Than or Equal Operators
constexpr Oid PG_NUMERIC_GE_OID = 1757;

// =============================================================================
// FUNCTION OIDs (
// =============================================================================

// Mathematical Functions
constexpr Oid PG_F_ABS_INT4 = 1397;
constexpr Oid PG_F_ABS_INT8 = 1398;
constexpr Oid PG_F_ABS_FLOAT4 = 1394;
constexpr Oid PG_F_ABS_FLOAT8 = 1395;
constexpr Oid PG_F_SQRT_FLOAT8 = 230;
constexpr Oid PG_F_POW_FLOAT8 = 232;
constexpr Oid PG_F_CEIL_FLOAT8 = 2308;
constexpr Oid PG_F_FLOOR_FLOAT8 = 2309;
constexpr Oid PG_F_ROUND_FLOAT8 = 233;

// String Functions
constexpr Oid PG_F_UPPER = 871;
constexpr Oid PG_F_LOWER = 870;
constexpr Oid PG_F_LENGTH = 1317;
constexpr Oid PG_F_SUBSTR = 877;
constexpr Oid PG_F_CONCAT = 3058;

// Date Functions
constexpr Oid PG_F_NOW = 3058;
constexpr Oid PG_F_DATE_PART = 230;
constexpr Oid PG_F_EXTRACT = 232;

// Type Conversion Functions
constexpr Oid PG_F_INT4_TEXT = 2308;
constexpr Oid PG_F_TEXT_INT4 = 2309;
constexpr Oid PG_F_FLOAT8_TEXT = 233;

// String Operator OIDs
constexpr Oid PG_TEXT_LIKE_OID = 1209;
constexpr Oid PG_TEXT_NOT_LIKE_OID = 1210;
constexpr Oid PG_TEXT_CONCAT_OID = 654;

// String Function OIDs
constexpr Oid PG_F_SUBSTRING = 936;
// Note: PG_F_UPPER and PG_F_LOWER already defined above

// Array Type OIDs
constexpr Oid PG_TEXT_ARRAY_OID = 1009;

// =============================================================================
// AGGREGATE FUNCTION OIDs
// =============================================================================

// Core aggregate functions (type-generic, we'll detect the base function)
constexpr Oid PG_F_SUM_INT2 = 1835;
constexpr Oid PG_F_SUM_INT4 = 2108;
constexpr Oid PG_F_SUM_INT8 = 2109;
constexpr Oid PG_F_SUM_FLOAT4 = 2110;
constexpr Oid PG_F_SUM_FLOAT8 = 2111;
constexpr Oid PG_F_SUM_NUMERIC = 2114;

constexpr Oid PG_F_AVG_INT2 = 1836;
constexpr Oid PG_F_AVG_INT4 = 2100;
constexpr Oid PG_F_AVG_INT8 = 2101;
constexpr Oid PG_F_AVG_FLOAT4 = 2102;
constexpr Oid PG_F_AVG_FLOAT8 = 2103;
constexpr Oid PG_F_AVG_NUMERIC = 2104;

constexpr Oid PG_F_COUNT_STAR = 2147;
constexpr Oid PG_F_COUNT_ANY = 2803;

constexpr Oid PG_F_MIN_INT2 = 2131;
constexpr Oid PG_F_MIN_INT4 = 2132;
constexpr Oid PG_F_MIN_INT8 = 2133;
constexpr Oid PG_F_MIN_FLOAT4 = 2135;
constexpr Oid PG_F_MIN_FLOAT8 = 2136;
constexpr Oid PG_F_MIN_NUMERIC = 2146;
constexpr Oid PG_F_MIN_TEXT = 2145;

constexpr Oid PG_F_MAX_INT2 = 2115;
constexpr Oid PG_F_MAX_INT4 = 2116;
constexpr Oid PG_F_MAX_INT8 = 2117;
constexpr Oid PG_F_MAX_FLOAT4 = 2119;
constexpr Oid PG_F_MAX_FLOAT8 = 2120;
constexpr Oid PG_F_MAX_NUMERIC = 2130;
constexpr Oid PG_F_MAX_TEXT = 2129;

// =============================================================================
// POSTGRESQL SYSTEM CONSTANTS (NOT TABLE NAMES - those should be dynamic!)
// =============================================================================

// PostgreSQL System Constants (defined after PostgreSQL headers are included)
constexpr Oid FIRST_NORMAL_OBJECT_ID = 16384;

// Node Type Constants (for unit test compatibility with lingo-db headers)
constexpr int LINGODB_T_VAR = 402;
constexpr int LINGODB_T_OPEXPR = 403;

// Boolean Expression Type Constants
constexpr int BOOL_AND_EXPR = 0;
constexpr int BOOL_OR_EXPR = 1;
constexpr int BOOL_NOT_EXPR = 2;

// PostgreSQL Null Test Constants
constexpr int PG_IS_NULL = 0;
constexpr int PG_IS_NOT_NULL = 1;

// Column naming patterns (only for generated columns, not schema assumptions)
constexpr const char* GENERATED_COLUMN_PREFIX = "col_";
constexpr const char* COMPUTED_EXPRESSION_SCOPE = "map";
constexpr const char* AGGREGATION_RESULT_COLUMN = "aggr_result";

// =============================================================================
// POSTGRESQL INTERNAL CONSTANTS
// =============================================================================

// PostgreSQL System Constants
constexpr int POSTGRESQL_VARNO_OFFSET = 1;
constexpr int POSTGRESQL_VARHDRSZ = 4;
constexpr int32_t INVALID_TYPMOD = -1;
constexpr int PG_LIST_INDEX_OFFSET = 1;

// PostgreSQL Column Attribute Numbers (1-based)
constexpr int FIRST_COLUMN_ATTNO = 1;
constexpr int SECOND_COLUMN_ATTNO = 2;
constexpr int THIRD_COLUMN_ATTNO = 3;
constexpr int INVALID_VARNO = 0;
constexpr int INVALID_VARATTNO = 0;

// =============================================================================
// POSTGRESQL TYPE SYSTEM CONSTANTS
// =============================================================================

// Numeric Type Processing
constexpr int MIN_NUMERIC_PRECISION = 1;
constexpr int DEFAULT_NUMERIC_SCALE = 0;
constexpr int NUMERIC_PRECISION_SHIFT = 16;
constexpr int NUMERIC_PRECISION_MASK = 0xFFFF;
constexpr int NUMERIC_SCALE_MASK = 0xFFFF;

// String Type Defaults
constexpr int DEFAULT_VARCHAR_LENGTH = 255;

constexpr double AVERAGE_DAYS_PER_MONTH = 29.53;

// Type Bit Widths (for MLIR type mapping)
constexpr int BOOL_BIT_WIDTH = 1;
constexpr int INT2_BIT_WIDTH = 16;
constexpr int INT4_BIT_WIDTH = 32;
constexpr int INT8_BIT_WIDTH = 64;

// Boolean Integer Values
constexpr int BOOL_TRUE_VALUE = 1;
constexpr int BOOL_FALSE_VALUE = 0;

// Timestamp Precision Levels
constexpr int TIMESTAMP_PRECISION_SECOND = 0;
constexpr int TIMESTAMP_PRECISION_MILLI_MIN = 1;
constexpr int TIMESTAMP_PRECISION_MILLI_MAX = 3;
constexpr int TIMESTAMP_PRECISION_MICRO_MIN = 4;
constexpr int TIMESTAMP_PRECISION_MICRO_MAX = 6;
constexpr int TIMESTAMP_PRECISION_NANO_MIN = 7;
constexpr int TIMESTAMP_PRECISION_NANO_MAX = 9;

// =============================================================================
// SORT OPERATOR OIDS
// =============================================================================

constexpr Oid PG_INT4_GE_ALT_OID = 523;
constexpr Oid PG_INT8_GE_ALT_OID = 525;

// =============================================================================
// EXPRESSION TRANSLATION CONSTANTS
// =============================================================================

// Expression Processing Constants
constexpr int MAX_BINARY_OPERANDS = 2;
constexpr int LEFT_OPERAND_INDEX = 0;
constexpr int RIGHT_OPERAND_INDEX = 1;
constexpr int MIN_ARGUMENT_COUNT = 1;
constexpr int UNARY_FUNCTION_ARGS = 1;
constexpr int BINARY_FUNCTION_ARGS = 2;

// Array and Result Indices
constexpr int FIRST_RESULT_INDEX = 0;
constexpr int FIRST_COLUMN_INDEX = 0;
constexpr int FIRST_LIST_INDEX = 0;
constexpr int FIRST_ATTRIBUTE_INDEX = 0;
constexpr int ARRAY_START_INDEX = 0;

// Default Placeholder Values
constexpr int DEFAULT_PLACEHOLDER_INT = 0;
constexpr int DEFAULT_PLACEHOLDER_BOOL = 1;
constexpr int DEFAULT_PLACEHOLDER_BOOL_FALSE = 0;
constexpr double DEFAULT_PLACEHOLDER_FLOAT = 0.0;
constexpr int DEFAULT_FALLBACK_INT_VALUE = 0;

// =============================================================================
// QUERY PROCESSING CONSTANTS
// =============================================================================

// Limit and Offset Constants
constexpr int64_t DEFAULT_LIMIT_OFFSET = 0;
constexpr int64_t UNLIMITED_LIMIT_VALUE = -1;
constexpr int64_t MAX_RESULT_ROWS = INT32_MAX;

// Validation Thresholds
constexpr int MIN_WORKER_COUNT = 0;
constexpr int INVALID_RELATION_ID = 0;
constexpr int INVALID_COLUMN_INDEX = 0;
constexpr int RELATION_ID_OFFSET = 1;
constexpr int SCANRELID_OFFSET = 1;
constexpr int MIN_COLUMN_COUNT = 0;
constexpr int MIN_COLUMN_INDEX = 0;
constexpr int MIN_LIST_LENGTH = 0;

// Table and Schema Constants
constexpr int INITIAL_ROW_COUNT = 0;
constexpr int FALLBACK_TABLE_OID = 0;
constexpr int EMPTY_SCHEMA_OID = 0;
constexpr int DEFAULT_ROW_COUNT = 0;
constexpr int INVALID_SCAN_RELID = 0;

// List Processing Constants
constexpr int EMPTY_LIST_LENGTH = 0;
constexpr int EMPTY_QUAL_LENGTH = 0;
constexpr int EMPTY_TARGET_LIST_LENGTH = 0;
constexpr int INVALID_TARGET_LIST_LENGTH = 0;

// =============================================================================
// STRING CONSTANTS
// =============================================================================

// Table and Column Prefixes
constexpr const char* UNIT_TEST_TABLE_PREFIX = "test_table_";
constexpr const char* FALLBACK_TABLE_PREFIX = "table_";
constexpr const char* DEFAULT_TABLE_NAME = "test";
constexpr const char* DEFAULT_COLUMN_NAME = "id";
constexpr const char* DEFAULT_TABLE_SCOPE = "test";

// Generated Names
constexpr const char* EXPRESSION_COLUMN_PREFIX = "expr_";
constexpr const char* TABLE_OID_SEPARATOR = "|oid:";
constexpr const char* QUERY_FUNCTION_NAME = "main";

// Aggregation Constants
constexpr const char* AGGREGATION_COUNT_FUNCTION = "count";
constexpr const char* AGGREGATION_SUM_FUNCTION = "sum";
constexpr const char* AGGREGATION_AVG_FUNCTION = "avg";
constexpr const char* AGGREGATION_MIN_FUNCTION = "min";
constexpr const char* AGGREGATION_MAX_FUNCTION = "max";

// Unit Test Column Configuration
constexpr bool UNIT_TEST_COLUMN_NOT_NULL = false;
constexpr bool PG_ATTNAME_NOT_MISSING_OK = false;

// =============================================================================
// CONFIGURATION LIMITS
// =============================================================================

constexpr int MAX_QUERY_COLUMNS = 1000;
constexpr int MAX_COLUMN_INDEX = 1000;
constexpr int MAX_LIST_LENGTH = 1000;
constexpr int MAX_NUMERIC_PRECISION = 1000;
constexpr int MAX_LIMIT_COUNT = 1000000;
constexpr int DEFAULT_LIMIT_COUNT = 10;

} // namespace pgx_lower::frontend::sql::constants

#endif // PGX_LOWER_CONSTANTS_H