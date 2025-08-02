LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → SubOp → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO WARNING;

-- Test 1: Simple text selection
DROP TABLE IF EXISTS simple_text;
CREATE TABLE simple_text (
    id INTEGER,
    txt TEXT
);
INSERT INTO simple_text VALUES (1, 'hello');
INSERT INTO simple_text VALUES (2, 'world');
SELECT id, txt FROM simple_text;

-- Test 2: CHAR column with padding
DROP TABLE IF EXISTS char_test;
CREATE TABLE char_test (
    id INTEGER,
    ch CHAR(10)
);
INSERT INTO char_test VALUES (1, 'abc');
INSERT INTO char_test VALUES (2, 'xyz');
SELECT id, ch FROM char_test;

-- Test 3: LPAD function
DROP TABLE IF EXISTS lpad_test;
CREATE TABLE lpad_test (
    id INTEGER,
    padded CHAR(10)
);
INSERT INTO lpad_test VALUES (1, LPAD('ch1', 10, 'x'));
INSERT INTO lpad_test VALUES (2, LPAD('ch2', 10, 'x'));
SELECT id, padded FROM lpad_test;