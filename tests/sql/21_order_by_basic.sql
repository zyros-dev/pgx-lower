-- Test basic ORDER BY functionality: ASC and DESC ordering
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → SubOp → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_order_basic;

CREATE TABLE test_order_basic (
    id SERIAL PRIMARY KEY,
    value INTEGER,
    name TEXT
);

INSERT INTO test_order_basic(value, name) VALUES 
    (30, 'Charlie'),
    (10, 'Alice'),
    (20, 'Bob'),
    (40, 'David'),
    (15, 'Eve');

-- Test basic ORDER BY ASC (default)
SELECT value, name FROM test_order_basic ORDER BY value;

-- Test explicit ORDER BY ASC  
SELECT value, name FROM test_order_basic ORDER BY value ASC;

-- Test ORDER BY DESC
SELECT value, name FROM test_order_basic ORDER BY value DESC;

-- Test ORDER BY on text column
SELECT value, name FROM test_order_basic ORDER BY name;

DROP TABLE test_order_basic;