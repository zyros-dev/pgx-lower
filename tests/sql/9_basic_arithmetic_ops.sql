LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_arithmetic;

CREATE TABLE test_arithmetic
(
    id   SERIAL PRIMARY KEY,
    val1 INTEGER,
    val2 INTEGER
);

INSERT INTO test_arithmetic(val1, val2)
VALUES (10, 5),
       (20, 4),
       (15, 3),
       (8, 2),
       (25, 6);

SET client_min_messages TO DEBUG1;
SET pgx_lower.log_enable = true;
SET pgx_lower.log_debug = true;
SET pgx_lower.log_io = true;
SET pgx_lower.log_ir = true;
SET pgx_lower.log_trace = true;
SET pgx_lower.log_verbose = true;
SET pgx_lower.enabled_categories = 'AST_TRANSLATE';
SELECT val1 + val2 AS addition
FROM test_arithmetic;
SET client_min_messages TO NOTICE;

SELECT val1 - val2 AS subtraction
FROM test_arithmetic;
SELECT val1 * val2 AS multiplication
FROM test_arithmetic;
SELECT val1 / val2 AS division
FROM test_arithmetic;
SELECT val1 % val2 AS modulo
FROM test_arithmetic;

DROP TABLE test_arithmetic;