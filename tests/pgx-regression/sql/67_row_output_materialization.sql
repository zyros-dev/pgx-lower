LOAD 'pgx_lower.so';

CREATE TEMP TABLE row_output_materialization(
    a int4,
    b int8 NOT NULL,
    c int4
);

INSERT INTO row_output_materialization VALUES
    (1, 100, NULL),
    (NULL, 200, 2),
    (7, 300, 3);

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_67_row_output_materialization_pset */
\pset format csv

SET pgx_lower.execution_mode = 'auto';
SET pgx_lower.route_path_notices = on;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_67_row_output_materialization_001 */
SELECT b, a FROM row_output_materialization WHERE b = 100;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_67_row_output_materialization_002 */
SELECT a, a IS NULL AS a_is_null FROM row_output_materialization WHERE b = 200;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_67_row_output_materialization_003 */
SELECT a AS dup, c AS dup FROM row_output_materialization WHERE b = 300;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_67_row_output_materialization_004 */
SELECT a, c, a = c AS a_eq_c FROM row_output_materialization WHERE b = 200;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=legacy id=pgx_67_row_output_materialization_005 */
SELECT a FROM row_output_materialization ORDER BY b;

SET pgx_lower.route_path_notices = off;

DROP TABLE row_output_materialization;
