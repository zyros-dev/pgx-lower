-- Test null handling operators: PgIsNullOp, PgIsNotNullOp, PgCoalesceOp
LOAD 'pgx_lower';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST  RelAlg  DB  LLVM  JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_nulls;

CREATE TABLE test_nulls (
    id SERIAL PRIMARY KEY,
    nullable_value INTEGER,
    backup_value INTEGER,
    third_value INTEGER
);

INSERT INTO test_nulls(nullable_value, backup_value, third_value) VALUES 
    (NULL, 100, 200),
    (10, 110, 210),
    (NULL, 120, 220),
    (30, NULL, 230),
    (40, 140, NULL);

-- Test null handling operations in SELECT clauses
-- These should trigger MLIR compilation with null handling operators
SELECT (nullable_value IS NULL) AS is_null_check FROM test_nulls;
SELECT (nullable_value IS NOT NULL) AS is_not_null_check FROM test_nulls;
SELECT (backup_value IS NULL) AS backup_is_null FROM test_nulls;
SELECT (backup_value IS NOT NULL) AS backup_is_not_null FROM test_nulls;
SELECT COALESCE(nullable_value, backup_value) AS coalesced_value FROM test_nulls;
SELECT COALESCE(nullable_value, backup_value, third_value) AS triple_coalesce FROM test_nulls;
SELECT COALESCE(nullable_value, -1) AS coalesce_with_constant FROM test_nulls;

DROP TABLE test_nulls;