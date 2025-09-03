LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id SERIAL
);

INSERT INTO test(id)
VALUES (42);

SELECT *
FROM test;
