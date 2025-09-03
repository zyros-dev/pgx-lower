LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id   SERIAL PRIMARY KEY,
    col2 BOOLEAN,
    col3 BOOLEAN,
    col4 BOOLEAN,
    col5 BOOLEAN
);

INSERT INTO test(col2, col3, col4, col5)
SELECT gs % 2 = 0,
    gs % 3 = 0,
    gs % 5 = 0,
    gs % 7 = 0
FROM generate_series(1, 100) AS gs;

SELECT col3, col5
FROM test;

SELECT col4
FROM test;
