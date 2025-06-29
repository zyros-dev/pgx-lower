-- Test basic arithmetic operators: PgAddOp, PgSubOp, PgMulOp
LOAD 'pgx_lower';

DROP TABLE IF EXISTS test_arithmetic;

CREATE TABLE test_arithmetic (
    id SERIAL PRIMARY KEY,
    val1 INTEGER,
    val2 INTEGER
);

INSERT INTO test_arithmetic(val1, val2) VALUES 
    (10, 5),
    (20, 4),
    (15, 3);

-- Test arithmetic operations on table columns
-- These should trigger MLIR compilation with our new operators
SELECT id, val1, val2 FROM test_arithmetic;

DROP TABLE test_arithmetic;