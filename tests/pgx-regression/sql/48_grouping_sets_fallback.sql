LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS grouping_sets_fallback;
CREATE TABLE grouping_sets_fallback(a int4);
INSERT INTO grouping_sets_fallback VALUES (1), (1), (2);

SET pgx_lower.execution_mode = 'auto';
\pset format unaligned
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_48_grouping_sets_fallback_001 */
SELECT a, count(*)
FROM grouping_sets_fallback
GROUP BY GROUPING SETS ((a), ())
ORDER BY a NULLS LAST;
\pset format aligned

DROP TABLE grouping_sets_fallback;
