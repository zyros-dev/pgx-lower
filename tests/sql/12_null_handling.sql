-- Test NULL handling operators: PgIsNullOp, PgIsNotNullOp
LOAD 'pgx_lower';

DROP TABLE IF EXISTS test_nulls;

CREATE TABLE test_nulls (
    id SERIAL PRIMARY KEY,
    nullable_value INTEGER,
    required_value INTEGER NOT NULL
);

INSERT INTO test_nulls(nullable_value, required_value) VALUES 
    (NULL, 1),
    (42, 2),
    (NULL, 3),
    (99, 4);

-- Test NULL handling operations on table columns
-- These should trigger MLIR compilation with NULL operators
SELECT id, nullable_value, required_value FROM test_nulls;

DROP TABLE test_nulls;