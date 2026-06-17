LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS aggdistinct_fallback;
CREATE TABLE aggdistinct_fallback(a int4, v float8);
INSERT INTO aggdistinct_fallback VALUES (1, 1.0), (1, 1.0), (2, 2.0), (2, 2.0);

SET pgx_lower.execution_mode = 'auto';
\pset format unaligned
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_52_aggdistinct_fallback_001 */
SELECT count(DISTINCT a) AS cda, sum(DISTINCT a) AS sda, sum(DISTINCT v) AS sdv
FROM aggdistinct_fallback;
\pset format aligned

DROP TABLE aggdistinct_fallback;
