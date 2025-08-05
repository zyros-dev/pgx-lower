LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;
SELECT 'hello';
DROP TABLE IF EXISTS test;
CREATE TABLE test(id SERIAL);
INSERT INTO test(id) VALUES (10);
INSERT INTO test(id) VALUES (1);
SELECT * FROM test;
