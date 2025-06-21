LOAD 'pgx_lower.so';
SELECT 'hello';
CREATE TABLE test(id SERIAL);
INSERT INTO test(id) VALUES (15);
INSERT INTO test(id) VALUES (16);
SELECT * FROM test;
