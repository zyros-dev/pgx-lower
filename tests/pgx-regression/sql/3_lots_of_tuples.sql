LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id SERIAL
);

INSERT INTO test(id)
SELECT generate_series(1, 5000);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_3_lots_of_tuples_001 */
SELECT *
FROM test;
