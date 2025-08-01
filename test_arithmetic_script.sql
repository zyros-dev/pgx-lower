-- Test script that mimics the original test
LOAD 'pgx_lower.so';
SET client_min_messages TO WARNING;
DROP TABLE IF EXISTS test_arithmetic_script;

CREATE TABLE test_arithmetic_script (
    id SERIAL PRIMARY KEY,
    val1 INTEGER,
    val2 INTEGER
);

INSERT INTO test_arithmetic_script(val1, val2) VALUES 
    (10, 5),
    (20, 4),
    (15, 3);

-- Test arithmetic operations
SELECT val1 + val2 AS addition FROM test_arithmetic_script;
SELECT val1 - val2 AS subtraction FROM test_arithmetic_script;

DROP TABLE test_arithmetic_script;