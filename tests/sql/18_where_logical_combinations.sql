-- Test WHERE clause with logical combinations: AND, OR, NOT
LOAD 'pgx_lower';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → SubOp → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_where_logical;

CREATE TABLE test_where_logical (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    score INTEGER,
    active BOOLEAN,
    department VARCHAR(20)
);

INSERT INTO test_where_logical(age, score, active, department) VALUES 
    (25, 85, true, 'Engineering'),
    (30, 92, false, 'Marketing'),
    (22, 78, true, 'Engineering'),
    (35, 88, true, 'Sales'),
    (28, 95, false, 'Marketing'),
    (32, 72, true, 'Sales'),
    (26, 89, false, 'Engineering');

-- Test WHERE with AND conditions
-- These should trigger MLIR compilation with logical AND operations
SELECT id, age, score FROM test_where_logical WHERE age > 25 AND score > 85;
SELECT department, age FROM test_where_logical WHERE active = true AND age < 30;
SELECT id, department FROM test_where_logical WHERE score >= 85 AND department = 'Engineering';

-- Test WHERE with OR conditions
SELECT id, age, score FROM test_where_logical WHERE age < 25 OR score > 90;
SELECT department, active FROM test_where_logical WHERE department = 'Sales' OR department = 'Marketing';
SELECT id, age FROM test_where_logical WHERE age > 35 OR active = false;

-- Test WHERE with NOT conditions
SELECT id, age, department FROM test_where_logical WHERE NOT active;
SELECT age, score FROM test_where_logical WHERE NOT (age < 25);
SELECT department FROM test_where_logical WHERE NOT (department = 'Engineering');

-- Test WHERE with complex logical combinations
SELECT id, age, score FROM test_where_logical WHERE (age > 25 AND score > 80) OR (active = false);
SELECT department, age FROM test_where_logical WHERE active = true AND (age < 30 OR score > 85);
SELECT id, department FROM test_where_logical WHERE NOT (age < 25 OR score < 80);

-- Test WHERE with multiple AND/OR combinations
SELECT age, score, department FROM test_where_logical WHERE age > 25 AND score > 80 AND active = true;
SELECT id, age FROM test_where_logical WHERE age < 30 OR score > 90 OR department = 'Sales';

DROP TABLE test_where_logical;