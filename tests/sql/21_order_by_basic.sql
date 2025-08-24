-- Test basic ORDER BY functionality: ASC and DESC ordering
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_order_basic;

CREATE TABLE test_order_basic (
    id SERIAL PRIMARY KEY,
    value INTEGER,
    name INTEGER    -- Replaces TEXT: 1=Alice, 2=Bob, 3=Charlie, 4=David, 5=Eve
);

-- Insert test data using only minimal supported types
INSERT INTO test_order_basic(value, name) VALUES 
    (30, 3),            -- Charlie = 3
    (10, 1),            -- Alice = 1
    (20, 2),            -- Bob = 2
    (40, 4),            -- David = 4
    (15, 5);            -- Eve = 5

-- Test basic ORDER BY ASC (default) on value column
SELECT value, name FROM test_order_basic ORDER BY value;

-- Test explicit ORDER BY ASC on value column
SELECT value, name FROM test_order_basic ORDER BY value ASC;

-- Test ORDER BY DESC on value column
SELECT value, name FROM test_order_basic ORDER BY value DESC;

-- Test ORDER BY on name column (equivalent to original text ordering)
SELECT value, name FROM test_order_basic ORDER BY name;

DROP TABLE test_order_basic;