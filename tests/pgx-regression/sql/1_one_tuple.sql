LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id SERIAL
);

INSERT INTO test(id)
VALUES (42);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_1_one_tuple_001 */
SELECT *
FROM test;
