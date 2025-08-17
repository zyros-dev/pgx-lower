LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test;

CREATE TABLE test (
    id SERIAL PRIMARY KEY,
    col2 BOOLEAN
);

INSERT INTO test(col2)
SELECT CASE WHEN gs % 2 = 0 THEN TRUE ELSE FALSE END
FROM generate_series(1, 100) AS gs;

SELECT * FROM test;