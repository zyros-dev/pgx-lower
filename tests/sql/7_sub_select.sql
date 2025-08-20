LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST  RelAlg  DB  LLVM  JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;
DROP TABLE IF EXISTS test;

CREATE TABLE test (
    id SERIAL PRIMARY KEY,
    col2 BOOLEAN,
    col3 BOOLEAN,
    col4 BOOLEAN,
    col5 BOOLEAN
);

INSERT INTO test(col2, col3, col4, col5)
SELECT
    gs % 2 = 0,
    gs % 3 = 0,
    gs % 5 = 0,
    gs % 7 = 0
FROM generate_series(1, 100) AS gs;

SELECT col3, col5 FROM test;

SELECT col4 FROM test;
