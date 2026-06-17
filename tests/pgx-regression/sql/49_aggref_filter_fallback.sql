LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS aggref_filter_fallback;
CREATE TABLE aggref_filter_fallback(a int4);
INSERT INTO aggref_filter_fallback VALUES (1), (1), (2);

SET pgx_lower.execution_mode = 'auto';
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_49_aggref_filter_fallback_001 */
SELECT count(*) FILTER (WHERE a = 1) AS filtered_count
FROM aggref_filter_fallback;

DROP TABLE aggref_filter_fallback;
