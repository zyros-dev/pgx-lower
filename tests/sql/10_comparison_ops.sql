-- Test comparison operators: PgCmpOp with all predicates (eq, ne, lt, le, gt, ge)
LOAD 'pgx_lower';
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_comparison;

CREATE TABLE test_comparison (
    id SERIAL PRIMARY KEY,
    value INTEGER,
    score INTEGER
);

INSERT INTO test_comparison(value, score) VALUES 
    (10, 15),
    (20, 20),
    (15, 10),
    (25, 5),
    (30, 30);

-- Test comparison operations in SELECT clauses
-- These should trigger MLIR compilation with comparison operators
SELECT (value = score) AS is_equal FROM test_comparison;
SELECT (value <> score) AS not_equal FROM test_comparison;
SELECT (value != score) AS not_equal_alt FROM test_comparison;
SELECT (value < score) AS less_than FROM test_comparison;
SELECT (value <= score) AS less_equal FROM test_comparison;
SELECT (value > score) AS greater_than FROM test_comparison;
SELECT (value >= score) AS greater_equal FROM test_comparison;

DROP TABLE test_comparison;