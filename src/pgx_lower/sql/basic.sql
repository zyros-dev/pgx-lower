LOAD 'pgx_lower.so';
SELECT 'hello';
CREATE TABLE test(id SERIAL);
INSERT INTO test(id) VALUES (2);
SELECT * FROM test;
