LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS expr_test_data;

CREATE TABLE expr_test_data (
    id INTEGER PRIMARY KEY,
    value1 INTEGER,
    value2 INTEGER,
    text1 VARCHAR(20),
    text2 VARCHAR(20),
    flag BOOLEAN
);

INSERT INTO expr_test_data (id, value1, value2, text1, text2, flag)
VALUES
    (1, 10, 5, 'hello', 'world', true),
    (2, 20, 15, 'test', 'data', false),
    (3, 30, 25, 'example', 'text', true),
    (4, 40, 35, 'sample', 'string', false),
    (5, 50, 45, 'demo', 'value', true);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_001 */
-- Test T_Integer in WHERE clause (direct integer constants)
SELECT id FROM expr_test_data WHERE 42 > 10;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_002 */
SELECT id FROM expr_test_data WHERE 100 = 100;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_003 */
-- Test T_A_Expr - arithmetic operations in SELECT
SELECT value1 + value2 AS sum_values FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_004 */
SELECT value1 - value2 AS diff_values FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_005 */
SELECT value1 * 2 AS double_value FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_006 */
SELECT value1 / 10 AS tenth_value FROM expr_test_data WHERE value1 > 0;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_30_test_missing_expressions_007 */
SELECT value1 % 7 AS modulo_result FROM expr_test_data;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_008 */
-- Test T_A_Expr - comparison operations in WHERE
SELECT id FROM expr_test_data WHERE value1 > 25;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_009 */
SELECT id FROM expr_test_data WHERE value2 < 30;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_010 */
SELECT id FROM expr_test_data WHERE value1 >= 30;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_011 */
SELECT id FROM expr_test_data WHERE value2 <= 25;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_012 */
SELECT id FROM expr_test_data WHERE value1 = 30;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_013 */
SELECT id FROM expr_test_data WHERE value1 != 20;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_014 */
SELECT id FROM expr_test_data WHERE value2 <> 15;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_015 */
-- Test T_A_Expr - logical operations
SELECT id FROM expr_test_data WHERE flag AND true;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_016 */
SELECT id FROM expr_test_data WHERE flag OR false;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_017 */
SELECT id FROM expr_test_data WHERE NOT flag;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_30_test_missing_expressions_018 */
-- Test T_A_Expr - string concatenation
SELECT text1 || ' ' || text2 AS combined_text FROM expr_test_data;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_019 */
-- Test T_A_Expr - LIKE operations
SELECT id, text1 FROM expr_test_data WHERE text1 LIKE 'h%';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_020 */
SELECT id, text1 FROM expr_test_data WHERE text1 NOT LIKE '%xyz%';

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_021 */
-- Test T_A_Expr - NULL operations
SELECT id FROM expr_test_data WHERE NULL IS NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_022 */
SELECT id FROM expr_test_data WHERE value1 IS NOT NULL;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_30_test_missing_expressions_023 */
-- Test T_A_Expr - IN operations
SELECT id FROM expr_test_data WHERE value1 IN (10, 20, 30);
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_30_test_missing_expressions_024 */
SELECT id FROM expr_test_data WHERE value1 NOT IN (15, 25, 35);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_025 */
-- Test T_A_Expr - BETWEEN operations
SELECT id FROM expr_test_data WHERE value1 BETWEEN 20 AND 40;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_026 */
SELECT id FROM expr_test_data WHERE value2 NOT BETWEEN 10 AND 20;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_027 */
-- Test T_A_Expr - Complex expressions
SELECT
    value1 + 10 AS plus_ten,
    value1 - 5 AS minus_five,
    value1 * value2 AS product,
    (value1 + value2) / 2 AS average
FROM expr_test_data
WHERE value1 > 10;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_028 */
-- Test combinations of expressions
SELECT
    id,
    value1 + value2 * 2 AS complex_calc,
    (value1 - 10) / 5 AS adjusted_value
FROM expr_test_data
WHERE (value1 > 20 AND value2 < 40) OR flag = true;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_029 */
-- Test T_ColumnRef - basic column references (already working but let's verify)
SELECT value1 FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_030 */
SELECT id, value1, value2 FROM expr_test_data WHERE id > 2;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_031 */
-- Test T_TypeCast - explicit type casting
SELECT CAST(value1 AS BIGINT) AS bigint_value FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_032 */
SELECT CAST(value1 AS DECIMAL(10,2)) AS decimal_value FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_30_test_missing_expressions_033 */
SELECT CAST(value1 AS VARCHAR(10)) AS string_value FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_034 */
SELECT value1::bigint AS bigint_cast FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_30_test_missing_expressions_035 */
SELECT value2::decimal(5,1) AS decimal_cast FROM expr_test_data;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_30_test_missing_expressions_036 */
SELECT id::text AS text_cast FROM expr_test_data;

-- Test T_ParamRef - parameter references (prepared statements)
-- These use $1, $2 style parameters
PREPARE test_param_stmt (int) AS
  SELECT id, value1 FROM expr_test_data WHERE value1 > $1;
EXECUTE test_param_stmt(25);
DEALLOCATE test_param_stmt;

PREPARE test_multi_param (int, int) AS
  SELECT id, value1, value2 FROM expr_test_data WHERE value1 > $1 AND value2 < $2;
EXECUTE test_multi_param(15, 30);
DEALLOCATE test_multi_param;


DROP TABLE expr_test_data;