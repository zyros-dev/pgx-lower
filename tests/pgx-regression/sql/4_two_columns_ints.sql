LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id   SERIAL PRIMARY KEY,
    col2 INTEGER
);

INSERT INTO test(col2)
SELECT generate_series(1, 100);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_4_two_columns_ints_001 */
SELECT *
FROM test;
