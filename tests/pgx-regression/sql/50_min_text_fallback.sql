LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS min_text_fallback;
CREATE TABLE min_text_fallback(name text);
INSERT INTO min_text_fallback VALUES ('delta'), ('alpha'), ('charlie');

SET pgx_lower.execution_mode = 'auto';
\pset format unaligned
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_50_min_text_fallback_001 */
SELECT min(name) AS min_name
FROM min_text_fallback;
\pset format aligned

DROP TABLE min_text_fallback;
