-- Test logical operators: PgAndOp, PgOrOp, PgNotOp
LOAD 'pgx_lower.so';
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_logical;

CREATE TABLE test_logical (
    id SERIAL PRIMARY KEY,
    flag1 BOOLEAN,
    flag2 BOOLEAN,
    value INTEGER
);

INSERT INTO test_logical(flag1, flag2, value) VALUES 
    (true, false, 10),
    (false, true, 20),
    (true, true, 30),
    (false, false, 40),
    (true, false, 50);

-- Test logical operations in SELECT clauses
-- These should trigger MLIR compilation with logical operators
SELECT (flag1 AND flag2) AS and_result FROM test_logical;
SELECT (flag1 OR flag2) AS or_result FROM test_logical;
SELECT (NOT flag1) AS not_flag1 FROM test_logical;
SELECT (NOT flag2) AS not_flag2 FROM test_logical;
SELECT (flag1 AND flag2 AND value > 25) AS complex_and FROM test_logical;
SELECT (flag1 OR flag2 OR value < 15) AS complex_or FROM test_logical;

DROP TABLE test_logical;