LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id SERIAL
);

INSERT INTO test(id)
VALUES (10);
INSERT INTO test(id)
VALUES (1);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_2_two_tuples_001 */
SELECT *
FROM test;
