#ifndef PGX_LOWER_CONSTANTS_H
#define PGX_LOWER_CONSTANTS_H

// PostgreSQL catalog headers should already be included by the main translator file

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
constexpr Oid PG_INT4_EQ_OID = 96;      // int4 = int4
constexpr Oid PG_INT8_EQ_OID = 410;     // int8 = int8

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
// EXPRESSION TRANSLATION CONSTANTS
// =============================================================================

// Expression Processing Constants
constexpr int MAX_BINARY_OPERANDS = 2;  // Maximum operands for binary operations
constexpr int LEFT_OPERAND_INDEX = 0;   // Left operand array index
constexpr int RIGHT_OPERAND_INDEX = 1;  // Right operand array index

// Default Placeholder Values
constexpr int DEFAULT_PLACEHOLDER_INT = 0;       // Default integer placeholder
constexpr int DEFAULT_PLACEHOLDER_BOOL = 1;      // Default boolean true
constexpr int DEFAULT_PLACEHOLDER_BOOL_FALSE = 0; // Default boolean false

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