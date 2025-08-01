LOAD 'pgx_lower.so';
CREATE TABLE debug_test (val1 INTEGER, val2 INTEGER);
INSERT INTO debug_test VALUES (10, 5);
SELECT val1 + val2 FROM debug_test;