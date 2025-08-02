LOAD 'pgx_lower.so';
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test;

CREATE TABLE test (
    id SERIAL PRIMARY KEY,
    col2 INTEGER
);

INSERT INTO test(col2)
SELECT generate_series(1, 100);

SELECT * FROM test LIMIT 5;
