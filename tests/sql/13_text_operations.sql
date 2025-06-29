-- Test text operations: PgTextAddOp, PgConcatOp, PgLikeOp, PgSubstringOp
LOAD 'pgx_lower';

DROP TABLE IF EXISTS test_text;

CREATE TABLE test_text (
    id SERIAL PRIMARY KEY,
    name TEXT,
    description TEXT
);

INSERT INTO test_text(name, description) VALUES 
    ('Alice', 'Software Engineer'),
    ('Bob', 'Data Scientist'),
    ('Charlie', 'Product Manager'),
    ('Diana', 'UX Designer');

-- Test text operations on table columns
-- These should trigger MLIR compilation with text operators
SELECT id, name, description FROM test_text;

DROP TABLE test_text;