LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST  RelAlg  DB  LLVM  JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

-- Test selecting just char column
DROP TABLE IF EXISTS char_only;
CREATE TABLE char_only (
    ch CHAR(10)
);
INSERT INTO char_only VALUES (LPAD('ch1', 10, 'x'));
INSERT INTO char_only VALUES (LPAD('ch2', 10, 'x'));
INSERT INTO char_only VALUES (LPAD('ch3', 10, 'x'));
SELECT ch FROM char_only;