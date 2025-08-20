-- Test basic arithmetic operators: PgAddOp, PgSubOp, PgMulOp, PgDivOp, PgModOp
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST  RelAlg  DB  LLVM  JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;
DROP TABLE IF EXISTS test_arithmetic;

CREATE TABLE test_arithmetic (
    id SERIAL PRIMARY KEY,
    val1 INTEGER,
    val2 INTEGER
);

INSERT INTO test_arithmetic(val1, val2) VALUES 
    (10, 5),
    (20, 4),
    (15, 3),
    (8, 2),
    (25, 6);

-- Test arithmetic operations in SELECT clauses
-- These should trigger MLIR compilation with arithmetic operators
SELECT val1 + val2 AS addition FROM test_arithmetic;
SELECT val1 - val2 AS subtraction FROM test_arithmetic;
SELECT val1 * val2 AS multiplication FROM test_arithmetic;
SELECT val1 / val2 AS division FROM test_arithmetic;
SELECT val1 % val2 AS modulo FROM test_arithmetic;

DROP TABLE test_arithmetic;