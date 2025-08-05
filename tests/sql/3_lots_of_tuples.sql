LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test;
CREATE TABLE test(id SERIAL);

-- Insert values 1 through 5000
INSERT INTO test(id)
SELECT generate_series(1, 1000000);

SELECT COUNT(*) FROM test;
