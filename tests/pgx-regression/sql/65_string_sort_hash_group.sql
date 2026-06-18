LOAD 'pgx_lower.so';

CREATE TEMP TABLE string_sort_hash_group(
    id int4,
    t text,
    v varchar(8),
    c3 char(3),
    c5 char(5)
);

INSERT INTO string_sort_hash_group VALUES
    (1, 'beta', 'beta', 'ab', 'ab'),
    (2, 'alpha', 'alpha', 'ab ', 'ab'),
    (3, 'gamma', 'gamma', 'xy', 'xy'),
    (4, 'alpha', 'alpha', 'xy ', 'xy');

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_65_string_sort_hash_group_pset */
\pset format csv

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_001 */
SELECT t FROM string_sort_hash_group ORDER BY t;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_002 */
SELECT c5, id FROM string_sort_hash_group ORDER BY c5, id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_003 */
SELECT c3, count(*) FROM string_sort_hash_group GROUP BY c3 ORDER BY c3;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_004 */
SELECT DISTINCT c3, 1 AS marker FROM string_sort_hash_group ORDER BY c3;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_005 */
SELECT a.id, b.id FROM string_sort_hash_group a JOIN string_sort_hash_group b ON a.c3 = b.c5 ORDER BY a.id, b.id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_006 */
SELECT min(t), max(t), min(c5), max(c5), count(*) AS row_count FROM string_sort_hash_group;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_007 */
SELECT v FROM string_sort_hash_group ORDER BY v;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_008 */
SELECT v, count(*) FROM string_sort_hash_group GROUP BY v ORDER BY v;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_65_string_sort_hash_group_009 */
SELECT min(v), max(v) FROM string_sort_hash_group;

DROP TABLE string_sort_hash_group;
