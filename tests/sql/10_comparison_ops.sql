-- Test comparison operators: PgCmpOp, PgNeOp, PgLtOp, PgLeOp, PgGtOp, PgGeOp
LOAD 'pgx_lower';

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
    (25, 5);

-- Test comparison operations on table columns
-- These should trigger MLIR compilation with comparison operators
SELECT id, value, score FROM test_comparison;

DROP TABLE test_comparison;