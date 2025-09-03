LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id SERIAL
);

INSERT INTO test(id)
SELECT generate_series(1, 5000);

SELECT *
FROM test;
