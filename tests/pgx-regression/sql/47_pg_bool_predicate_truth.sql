LOAD 'pgx_lower.so';

CREATE TEMP TABLE pgx_bool_truth_source(id int, flag boolean);
INSERT INTO pgx_bool_truth_source VALUES (1, true), (2, false), (3, NULL);
CREATE TEMP TABLE pgx_bool_truth_marker(marker text);
INSERT INTO pgx_bool_truth_marker VALUES ('joined');

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_47_pg_bool_predicate_truth_pset */
\pset format unaligned

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_47_pg_bool_predicate_truth_001 */
SELECT id FROM pgx_bool_truth_source WHERE flag ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_47_pg_bool_predicate_truth_002 */
SELECT id FROM pgx_bool_truth_source WHERE flag IS NULL ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_47_pg_bool_predicate_truth_003 */
SELECT s.id, m.marker
FROM pgx_bool_truth_source s
JOIN pgx_bool_truth_marker m ON s.flag
ORDER BY s.id, m.marker;

DROP TABLE pgx_bool_truth_marker;
DROP TABLE pgx_bool_truth_source;
