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

SET client_min_messages TO DEBUG1;
SET pgx_lower.log_enable = true;
SET pgx_lower.log_debug = true;
SET pgx_lower.log_io = true;
SET pgx_lower.log_ir = true;
SET pgx_lower.log_trace = true;
SET pgx_lower.log_verbose = true;
SET pgx_lower.enabled_categories = 'AST_TRANSLATE,RELALG_LOWER,DB_LOWER,DSA_LOWER,UTIL_LOWER,RUNTIME,JIT,GENERAL';

SELECT value1::bigint AS bigint_cast FROM expr_test_data;
SELECT value2::decimal(5,1) AS decimal_cast FROM expr_test_data;
SELECT id::text AS text_cast FROM expr_test_data;
SELECT CAST(value1 AS BIGINT) AS bigint_value FROM expr_test_data;
SELECT CAST(value1 AS DECIMAL(10,2)) AS decimal_value FROM expr_test_data;
SELECT CAST(value1 AS VARCHAR(10)) AS string_value FROM expr_test_data;
