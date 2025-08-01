LOAD 'pgx_lower.so';
DROP TABLE IF EXISTS test_arithmetic;
CREATE TABLE test_arithmetic (
    id SERIAL PRIMARY KEY,
    val1 INTEGER,
    val2 INTEGER
);
INSERT INTO test_arithmetic(val1, val2) VALUES (10, 5);
SELECT val1 + val2 AS addition FROM test_arithmetic;