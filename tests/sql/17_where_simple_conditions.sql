-- Test WHERE clause with simple conditions: equality, inequality, comparison operators
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_where_simple;

CREATE TABLE test_where_simple (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    score INTEGER,
    name INTEGER      -- Replaces VARCHAR(50): 1=Alice, 2=Bob, 3=Carol, 4=David, 5=Eve
);

-- Insert test data using only minimal supported types
INSERT INTO test_where_simple(age, score, name) VALUES 
    (25, 85, 1),          -- Alice = 1
    (30, 92, 2),          -- Bob = 2  
    (22, 78, 3),          -- Carol = 3
    (35, 88, 4),          -- David = 4
    (28, 95, 5);          -- Eve = 5

-- Test WHERE with equality conditions
-- These should trigger MLIR compilation with WHERE filtering
SELECT name, age FROM test_where_simple WHERE age = 25;
SELECT name, score FROM test_where_simple WHERE score = 92;
SELECT id, name FROM test_where_simple WHERE name = 3;

-- Test WHERE with inequality conditions
SELECT name, age FROM test_where_simple WHERE age <> 25;
SELECT name, score FROM test_where_simple WHERE score != 88;

-- Test WHERE with comparison conditions
SELECT name, age FROM test_where_simple WHERE age > 25;
SELECT name, age FROM test_where_simple WHERE age >= 30;
SELECT name, score FROM test_where_simple WHERE score < 90;
SELECT name, score FROM test_where_simple WHERE score <= 85;

-- Test WHERE with multiple simple conditions (one at a time)
SELECT name FROM test_where_simple WHERE age > 20;
SELECT name FROM test_where_simple WHERE score >= 85;

DROP TABLE test_where_simple;