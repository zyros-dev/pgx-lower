-- Test ORDER BY with multiple columns: various combinations of ASC/DESC ordering
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST  RelAlg  DB  LLVM  JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_order_multi;

CREATE TABLE test_order_multi (
    id SERIAL PRIMARY KEY,
    department VARCHAR(20),
    salary INTEGER,
    years INTEGER,
    name VARCHAR(50)
);

INSERT INTO test_order_multi(department, salary, years, name) VALUES 
    ('Sales', 50000, 3, 'Alice'),
    ('IT', 60000, 2, 'Bob'),
    ('Sales', 45000, 5, 'Carol'),
    ('IT', 65000, 4, 'David'),
    ('Sales', 50000, 1, 'Eve'),
    ('IT', 60000, 6, 'Frank'),
    ('HR', 55000, 3, 'Grace');

-- Test ORDER BY multiple columns, both ASC (default)
SELECT department, salary, name FROM test_order_multi ORDER BY department, salary;

-- Test ORDER BY multiple columns, explicit ASC
SELECT department, salary, name FROM test_order_multi ORDER BY department ASC, salary ASC;

-- Test ORDER BY multiple columns, mixed ASC/DESC
SELECT department, salary, name FROM test_order_multi ORDER BY department ASC, salary DESC;

-- Test ORDER BY multiple columns, both DESC
SELECT department, salary, name FROM test_order_multi ORDER BY department DESC, salary DESC;

-- Test ORDER BY three columns
SELECT department, salary, years, name FROM test_order_multi ORDER BY department, salary, years;

-- Test ORDER BY with complex mixed ordering
SELECT department, salary, years, name FROM test_order_multi ORDER BY department ASC, salary DESC, years ASC;

DROP TABLE test_order_multi;