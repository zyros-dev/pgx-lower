-- Test text operators: PgLikeOp, PgConcatOp, PgSubstringOp
LOAD 'pgx_lower';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST  RelAlg  DB  LLVM  JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_text;

CREATE TABLE test_text (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    description TEXT,
    category VARCHAR(20)
);

INSERT INTO test_text(name, description, category) VALUES 
    ('Apple', 'Fresh red apple fruit', 'fruit'),
    ('Banana', 'Yellow curved banana', 'fruit'),
    ('Carrot', 'Orange root vegetable', 'vegetable'),
    ('Avocado', 'Green creamy avocado', 'fruit'),
    ('Spinach', 'Green leafy vegetable', 'vegetable');

-- Test text operations in SELECT clauses
-- These should trigger MLIR compilation with text operators
SELECT (name LIKE 'A%') AS starts_with_a FROM test_text;
SELECT (name LIKE '%a%') AS contains_a FROM test_text;
SELECT (description LIKE '%green%') AS contains_green FROM test_text;
SELECT (category LIKE 'fruit') AS is_fruit FROM test_text;
SELECT (name || ' - ' || description) AS concatenated FROM test_text;
SELECT (name || ' (' || category || ')') AS name_with_category FROM test_text;
SELECT SUBSTRING(name FROM 1 FOR 3) AS name_prefix FROM test_text;
SELECT SUBSTRING(description FROM 1 FOR 10) AS desc_start FROM test_text;
SELECT UPPER(name) AS upper_name FROM test_text;
SELECT LOWER(description) AS lower_desc FROM test_text;

DROP TABLE test_text;