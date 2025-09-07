#ifndef PGX_LOWER_CONSTANTS_H
#define PGX_LOWER_CONSTANTS_H

// PostgreSQL catalog headers should already be included by the main translator file
// TODO: When we have this many constants, maybe we should create like PlusOid<data_type>()
// rather than a gazillion constants

namespace pgx_lower::frontend::sql::constants {

// =============================================================================
// ARITHMETIC OPERATOR OIDS
// =============================================================================

// Integer Addition Operators
constexpr Oid PG_INT4_PLUS_OID = 551;   // int4 + int4
constexpr Oid PG_INT8_PLUS_OID = 684;   // int8 + int8

// Integer Subtraction Operators  
constexpr Oid PG_INT4_MINUS_OID = 552;  // int4 - int4
constexpr Oid PG_INT4_MINUS_ALT_OID = 555; // Alternative int4 - int4
constexpr Oid PG_INT8_MINUS_OID = 688;  // int8 - int8

// Integer Multiplication Operators
constexpr Oid PG_INT4_MUL_OID = 514;    // int4 * int4
constexpr Oid PG_INT8_MUL_OID = 686;    // int8 * int8

// Integer Division Operators
constexpr Oid PG_INT4_DIV_OID = 527;    // int4 / int4 (alternative)
constexpr Oid PG_INT4_DIV_ALT_OID = 528; // int4 / int4
constexpr Oid PG_INT8_DIV_OID = 689;    // int8 / int8

// Integer Modulo Operators
constexpr Oid PG_INT4_MOD_OID = 529;    // int4 % int4
constexpr Oid PG_INT4_MOD_ALT_OID = 530; // int4 % int4 (alternative)
constexpr Oid PG_INT8_MOD_OID = 690;    // int8 % int8

// =============================================================================
// COMPARISON OPERATOR OIDS
// =============================================================================

// Integer Equality Operators
constexpr Oid PG_INT2_EQ_OID = 92;      // int2 = int2
constexpr Oid PG_INT4_EQ_OID = 96;      // int4 = int4
constexpr Oid PG_INT8_EQ_OID = 410;     // int8 = int8
constexpr Oid PG_TEXT_EQ_OID = 98;      // text = text

// Integer Inequality Operators
constexpr Oid PG_INT4_NE_OID = 518;     // int4 != int4
constexpr Oid PG_INT8_NE_OID = 411;     // int8 != int8

// Integer Less Than Operators
constexpr Oid PG_INT4_LT_OID = 97;      // int4 < int4
constexpr Oid PG_INT8_LT_OID = 412;     // int8 < int8

// Integer Greater Than Operators
constexpr Oid PG_INT4_GT_OID = 521;     // int4 > int4
constexpr Oid PG_INT8_GT_OID = 413;     // int8 > int8

// Integer Less Than or Equal Operators
constexpr Oid PG_INT4_LE_OID = 523;     // int4 <= int4
constexpr Oid PG_INT8_LE_OID = 414;     // int8 <= int8

// Integer Greater Than or Equal Operators
constexpr Oid PG_INT4_GE_OID = 525;     // int4 >= int4
constexpr Oid PG_INT8_GE_OID = 415;     // int8 >= int8

// =============================================================================
// FUNCTION OIDS (Complete mapping for expression translation)
// =============================================================================

// Mathematical Functions
constexpr Oid PG_F_ABS_INT4 = 1397;     // abs(int4)
constexpr Oid PG_F_ABS_INT8 = 1398;     // abs(int8)
constexpr Oid PG_F_ABS_FLOAT4 = 1394;   // abs(float4)
constexpr Oid PG_F_ABS_FLOAT8 = 1395;   // abs(float8)
constexpr Oid PG_F_SQRT_FLOAT8 = 230;   // sqrt(float8)
constexpr Oid PG_F_POW_FLOAT8 = 232;    // power(float8, float8)
constexpr Oid PG_F_CEIL_FLOAT8 = 2308;  // ceil(float8)
constexpr Oid PG_F_FLOOR_FLOAT8 = 2309; // floor(float8)
constexpr Oid PG_F_ROUND_FLOAT8 = 233;  // round(float8)

// String Functions
constexpr Oid PG_F_UPPER = 871;         // upper(text)
constexpr Oid PG_F_LOWER = 870;         // lower(text)
constexpr Oid PG_F_LENGTH = 1317;       // length(text)
constexpr Oid PG_F_SUBSTR = 877;        // substr(text, int4, int4)
constexpr Oid PG_F_CONCAT = 3058;       // concat(text, text)

// Date Functions  
constexpr Oid PG_F_NOW = 3058;          // now()
constexpr Oid PG_F_DATE_PART = 230;     // date_part(text, timestamp)
constexpr Oid PG_F_EXTRACT = 232;       // extract(field from timestamp)

// Type Conversion Functions
constexpr Oid PG_F_INT4_TEXT = 2308;    // int4 to text
constexpr Oid PG_F_TEXT_INT4 = 2309;    // text to int4
constexpr Oid PG_F_FLOAT8_TEXT = 233;   // float8 to text

// String Operator OIDs
constexpr Oid PG_TEXT_LIKE_OID = 1209;     // text LIKE text
constexpr Oid PG_TEXT_CONCAT_OID = 654;    // text || text

// String Function OIDs  
constexpr Oid PG_F_SUBSTRING = 936;        // substring(text, int, int)
// Note: PG_F_UPPER and PG_F_LOWER already defined above

// =============================================================================
// AGGREGATE FUNCTION OIDS
// =============================================================================

// Core aggregate functions (type-generic, we'll detect the base function)
constexpr Oid PG_F_SUM_INT2 = 1835;        // sum(int2) -> int8
constexpr Oid PG_F_SUM_INT4 = 2108;        // sum(int4) -> int8
constexpr Oid PG_F_SUM_INT8 = 2109;        // sum(int8) -> numeric
constexpr Oid PG_F_SUM_FLOAT4 = 2110;      // sum(float4) -> float4
constexpr Oid PG_F_SUM_FLOAT8 = 2111;      // sum(float8) -> float8
constexpr Oid PG_F_SUM_NUMERIC = 2114;     // sum(numeric) -> numeric

constexpr Oid PG_F_AVG_INT2 = 1836;        // avg(int2) -> numeric
constexpr Oid PG_F_AVG_INT4 = 2100;        // avg(int4) -> numeric
constexpr Oid PG_F_AVG_INT8 = 2101;        // avg(int8) -> numeric
constexpr Oid PG_F_AVG_FLOAT4 = 2102;      // avg(float4) -> float4
constexpr Oid PG_F_AVG_FLOAT8 = 2103;      // avg(float8) -> float8
constexpr Oid PG_F_AVG_NUMERIC = 2104;     // avg(numeric) -> numeric

constexpr Oid PG_F_COUNT_STAR = 2147;      // count(*) -> int8
constexpr Oid PG_F_COUNT_ANY = 2803;       // count(any) -> int8

constexpr Oid PG_F_MIN_INT2 = 2131;        // min(int2) -> int2
constexpr Oid PG_F_MIN_INT4 = 2132;        // min(int4) -> int4
constexpr Oid PG_F_MIN_INT8 = 2133;        // min(int8) -> int8
constexpr Oid PG_F_MIN_FLOAT4 = 2135;      // min(float4) -> float4
constexpr Oid PG_F_MIN_FLOAT8 = 2136;      // min(float8) -> float8
constexpr Oid PG_F_MIN_NUMERIC = 2146;     // min(numeric) -> numeric
constexpr Oid PG_F_MIN_TEXT = 2145;        // min(text) -> text

constexpr Oid PG_F_MAX_INT2 = 2115;        // max(int2) -> int2
constexpr Oid PG_F_MAX_INT4 = 2116;        // max(int4) -> int4
constexpr Oid PG_F_MAX_INT8 = 2117;        // max(int8) -> int8
constexpr Oid PG_F_MAX_FLOAT4 = 2119;      // max(float4) -> float4
constexpr Oid PG_F_MAX_FLOAT8 = 2120;      // max(float8) -> float8
constexpr Oid PG_F_MAX_NUMERIC = 2130;     // max(numeric) -> numeric
constexpr Oid PG_F_MAX_TEXT = 2129;        // max(text) -> text

// =============================================================================
// POSTGRESQL SYSTEM CONSTANTS (NOT TABLE NAMES - those should be dynamic!)
// =============================================================================

// PostgreSQL System Constants (defined after PostgreSQL headers are included)
constexpr Oid FIRST_NORMAL_OBJECT_ID = 16384; // FirstNormalObjectId - defined explicitly to avoid header dependency

// Node Type Constants (for unit test compatibility with lingo-db headers)
constexpr int LINGODB_T_VAR = 402;      // T_Var from lingo-db headers (unit tests)
constexpr int LINGODB_T_OPEXPR = 403;   // T_OpExpr from lingo-db headers (unit tests)

// Boolean Expression Type Constants
constexpr int BOOL_AND_EXPR = 0;        // BoolExprType AND
constexpr int BOOL_OR_EXPR = 1;         // BoolExprType OR  
constexpr int BOOL_NOT_EXPR = 2;        // BoolExprType NOT

// PostgreSQL Null Test Constants
constexpr int PG_IS_NULL = 0;           // IS NULL test
constexpr int PG_IS_NOT_NULL = 1;       // IS NOT NULL test

// Column naming patterns (only for generated columns, not schema assumptions)
constexpr const char* GENERATED_COLUMN_PREFIX = "col_";
constexpr const char* COMPUTED_EXPRESSION_SCOPE = "map";
constexpr const char* AGGREGATION_RESULT_COLUMN = "aggr_result";

// =============================================================================
// POSTGRESQL INTERNAL CONSTANTS
// =============================================================================

// PostgreSQL System Constants
constexpr int POSTGRESQL_VARNO_OFFSET = 1;          // 1-based to 0-based varno conversion
constexpr int POSTGRESQL_VARHDRSZ = 4;              // PostgreSQL variable header size  
constexpr int32_t INVALID_TYPMOD = -1;              // Invalid/unspecified typmod
constexpr int PG_LIST_INDEX_OFFSET = 1;             // PostgreSQL list index adjustment

// PostgreSQL Column Attribute Numbers (1-based)
constexpr int FIRST_COLUMN_ATTNO = 1;               // First column attribute number
constexpr int SECOND_COLUMN_ATTNO = 2;              // Second column attribute number  
constexpr int THIRD_COLUMN_ATTNO = 3;               // Third column attribute number
constexpr int INVALID_VARNO = 0;                    // Invalid range table entry number
constexpr int INVALID_VARATTNO = 0;                 // Invalid column attribute number

// =============================================================================
// POSTGRESQL TYPE SYSTEM CONSTANTS
// =============================================================================

// Numeric Type Processing
constexpr int MIN_NUMERIC_PRECISION = 1;            // Minimum NUMERIC precision
constexpr int DEFAULT_NUMERIC_SCALE = 0;            // Default NUMERIC scale
constexpr int NUMERIC_PRECISION_SHIFT = 16;         // Bit shift for precision extraction
constexpr int NUMERIC_PRECISION_MASK = 0xFFFF;      // 16-bit mask for precision
constexpr int NUMERIC_SCALE_MASK = 0xFFFF;          // 16-bit mask for scale

// String Type Defaults
constexpr int DEFAULT_VARCHAR_LENGTH = 255;         // Default VARCHAR length when unspecified

// Type Bit Widths (for MLIR type mapping)
constexpr int INT2_BIT_WIDTH = 16;                  // PostgreSQL INT2/SMALLINT bit width
constexpr int INT4_BIT_WIDTH = 32;                  // PostgreSQL INT4/INTEGER bit width  
constexpr int INT8_BIT_WIDTH = 64;                  // PostgreSQL INT8/BIGINT bit width
constexpr int BOOL_BIT_WIDTH = 1;                   // PostgreSQL BOOLEAN bit width

// Boolean Integer Values
constexpr int BOOL_TRUE_VALUE = 1;                  // Boolean true as integer
constexpr int BOOL_FALSE_VALUE = 0;                 // Boolean false as integer

// Timestamp Precision Levels
constexpr int TIMESTAMP_PRECISION_SECOND = 0;       // Second precision
constexpr int TIMESTAMP_PRECISION_MILLI_MIN = 1;    // Millisecond min precision
constexpr int TIMESTAMP_PRECISION_MILLI_MAX = 3;    // Millisecond max precision
constexpr int TIMESTAMP_PRECISION_MICRO_MIN = 4;    // Microsecond min precision  
constexpr int TIMESTAMP_PRECISION_MICRO_MAX = 6;    // Microsecond max precision
constexpr int TIMESTAMP_PRECISION_NANO_MIN = 7;     // Nanosecond min precision
constexpr int TIMESTAMP_PRECISION_NANO_MAX = 9;     // Nanosecond max precision

// =============================================================================
// SORT OPERATOR OIDS  
// =============================================================================

constexpr Oid PG_INT4_GE_ALT_OID = 523;            // int4 >= int4 (alternative)
constexpr Oid PG_INT8_GE_ALT_OID = 525;            // int8 >= int8 (alternative)

// =============================================================================
// EXPRESSION TRANSLATION CONSTANTS
// =============================================================================

// Expression Processing Constants
constexpr int MAX_BINARY_OPERANDS = 2;              // Maximum operands for binary operations
constexpr int LEFT_OPERAND_INDEX = 0;               // Left operand array index
constexpr int RIGHT_OPERAND_INDEX = 1;              // Right operand array index
constexpr int MIN_ARGUMENT_COUNT = 1;               // Minimum function argument count
constexpr int UNARY_FUNCTION_ARGS = 1;              // Unary function argument count
constexpr int BINARY_FUNCTION_ARGS = 2;             // Binary function argument count

// Array and Result Indices
constexpr int FIRST_RESULT_INDEX = 0;               // First result from operation
constexpr int FIRST_COLUMN_INDEX = 0;               // First column in iteration
constexpr int FIRST_LIST_INDEX = 0;                 // First element in list iteration
constexpr int FIRST_ATTRIBUTE_INDEX = 0;            // First attribute in tuple descriptor
constexpr int ARRAY_START_INDEX = 0;                // Generic array start index

// Default Placeholder Values
constexpr int DEFAULT_PLACEHOLDER_INT = 0;          // Default integer placeholder
constexpr int DEFAULT_PLACEHOLDER_BOOL = 1;         // Default boolean true
constexpr int DEFAULT_PLACEHOLDER_BOOL_FALSE = 0;   // Default boolean false
constexpr double DEFAULT_PLACEHOLDER_FLOAT = 0.0;   // Default float placeholder
constexpr int DEFAULT_FALLBACK_INT_VALUE = 0;       // Default integer fallback

// =============================================================================
// QUERY PROCESSING CONSTANTS
// =============================================================================

// Limit and Offset Constants
constexpr int64_t DEFAULT_LIMIT_OFFSET = 0;         // Default query offset
constexpr int64_t UNLIMITED_LIMIT_VALUE = -1;       // Special value for "no limit"
constexpr int64_t MAX_RESULT_ROWS = INT32_MAX;      // Maximum rows when no limit specified

// Validation Thresholds
constexpr int MIN_WORKER_COUNT = 0;                 // Minimum parallel worker count
constexpr int INVALID_RELATION_ID = 0;              // Invalid relation identifier
constexpr int INVALID_COLUMN_INDEX = 0;             // Invalid column index value
constexpr int RELATION_ID_OFFSET = 1;               // Offset for relation ID calculations
constexpr int SCANRELID_OFFSET = 1;                 // Scan relation ID adjustment
constexpr int MIN_COLUMN_COUNT = 0;                 // Minimum column count threshold
constexpr int MIN_COLUMN_INDEX = 0;                 // Minimum valid column index
constexpr int MIN_LIST_LENGTH = 0;                  // Minimum list length

// Table and Schema Constants
constexpr int INITIAL_ROW_COUNT = 0;                // Initial row count for tables
constexpr int FALLBACK_TABLE_OID = 0;               // Fallback OID when table not found
constexpr int EMPTY_SCHEMA_OID = 0;                 // OID for empty/missing schema
constexpr int DEFAULT_ROW_COUNT = 0;                // Default row count
constexpr int INVALID_SCAN_RELID = 0;               // Invalid scan relation identifier

// List Processing Constants
constexpr int EMPTY_LIST_LENGTH = 0;                // Empty list indicator
constexpr int EMPTY_QUAL_LENGTH = 0;                // Empty qualifier list
constexpr int EMPTY_TARGET_LIST_LENGTH = 0;         // Empty target list
constexpr int INVALID_TARGET_LIST_LENGTH = 0;       // Invalid target list length

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