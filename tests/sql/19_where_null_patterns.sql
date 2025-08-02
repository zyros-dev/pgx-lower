-- Test WHERE clause with NULL handling: IS NULL, IS NOT NULL, NULL comparisons
LOAD 'pgx_lower';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → SubOp → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_where_nulls;

CREATE TABLE test_where_nulls (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    age INTEGER,
    email VARCHAR(100),
    score INTEGER
);

INSERT INTO test_where_nulls(name, age, email, score) VALUES 
    ('Alice', 25, 'alice@test.com', 85),
    ('Bob', NULL, 'bob@test.com', 92),
    ('Carol', 30, NULL, 78),
    ('David', 35, 'david@test.com', NULL),
    ('Eve', NULL, NULL, 95),
    ('Frank', 28, 'frank@test.com', NULL),
    ('Grace', 32, NULL, 88);

-- Test WHERE with IS NULL conditions
-- These should trigger MLIR compilation with NULL checking operations
SELECT id, name, age FROM test_where_nulls WHERE age IS NULL;
SELECT id, name, email FROM test_where_nulls WHERE email IS NULL;
SELECT id, name, score FROM test_where_nulls WHERE score IS NULL;

-- Test WHERE with IS NOT NULL conditions
SELECT id, name, age FROM test_where_nulls WHERE age IS NOT NULL;
SELECT id, name, email FROM test_where_nulls WHERE email IS NOT NULL;
SELECT id, name, score FROM test_where_nulls WHERE score IS NOT NULL;

-- Test WHERE with NULL-safe comparisons
-- Note: NULL = NULL should return NULL (unknown), not true
SELECT id, name FROM test_where_nulls WHERE age = NULL;  -- Should return no rows
SELECT id, name FROM test_where_nulls WHERE email <> NULL;  -- Should return no rows

-- Test WHERE with NULL and logical combinations
SELECT id, name, age FROM test_where_nulls WHERE age IS NULL OR age < 30;
SELECT id, name, email FROM test_where_nulls WHERE email IS NOT NULL AND age > 25;
SELECT id, name FROM test_where_nulls WHERE score IS NULL AND age IS NOT NULL;

-- Test WHERE with COALESCE and NULL handling
SELECT id, name, age FROM test_where_nulls WHERE COALESCE(age, 0) > 25;
SELECT id, name, score FROM test_where_nulls WHERE COALESCE(score, 0) >= 85;

-- Test WHERE with multiple NULL conditions
SELECT id, name FROM test_where_nulls WHERE age IS NULL AND email IS NULL;
SELECT id, name FROM test_where_nulls WHERE age IS NOT NULL AND email IS NOT NULL AND score IS NOT NULL;
SELECT id, name FROM test_where_nulls WHERE age IS NULL OR email IS NULL OR score IS NULL;

-- Test WHERE with NULL in complex expressions
SELECT id, name, age FROM test_where_nulls WHERE (age IS NULL) OR (age > 30 AND score IS NOT NULL);
SELECT id, name FROM test_where_nulls WHERE NOT (age IS NULL);

DROP TABLE test_where_nulls;