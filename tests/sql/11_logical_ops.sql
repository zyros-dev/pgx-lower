-- Test logical operators: PgAndOp, PgOrOp, PgNotOp
LOAD 'pgx_lower';

DROP TABLE IF EXISTS test_logical;

CREATE TABLE test_logical (
    id SERIAL PRIMARY KEY,
    flag1 BOOLEAN,
    flag2 BOOLEAN
);

INSERT INTO test_logical(flag1, flag2) VALUES 
    (true, false),
    (false, true),
    (true, true),
    (false, false);

-- Test logical operations on table columns
-- These should trigger MLIR compilation with logical operators
SELECT id, flag1, flag2 FROM test_logical;

DROP TABLE test_logical;