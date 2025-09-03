LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id   SERIAL PRIMARY KEY,
    col2 BOOLEAN
);

INSERT INTO test(col2)
SELECT CASE WHEN gs % 2 = 0 THEN TRUE ELSE FALSE END
FROM generate_series(1, 100) AS gs;

SELECT *
FROM test;