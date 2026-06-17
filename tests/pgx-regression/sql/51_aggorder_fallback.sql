LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS aggorder_fallback;
CREATE TABLE aggorder_fallback(k int4, v float8);
INSERT INTO aggorder_fallback VALUES (1, 1e16::float8), (3, 1::float8), (2, -1e16::float8);

SET pgx_lower.execution_mode = 'auto';
\pset format unaligned
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_51_aggorder_fallback_001 */
SELECT sum(v ORDER BY k) AS asc_sum, sum(v ORDER BY k DESC) AS desc_sum
FROM aggorder_fallback;
\pset format aligned

DROP TABLE aggorder_fallback;
