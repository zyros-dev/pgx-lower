LOAD 'pgx_lower.so';
SELECT 'hello';
DROP TABLE IF EXISTS test;
CREATE TABLE test(id SERIAL);
INSERT INTO test(id) VALUES (10);
INSERT INTO test(id) VALUES (1);
SELECT * FROM test;
