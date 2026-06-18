LOAD 'pgx_lower.so';

CREATE TEMP TABLE row_first_slice(
    id int8 NOT NULL,
    payload int4
);

INSERT INTO row_first_slice VALUES
    (1, 10),
    (7, 20),
    (12, 30),
    (13, NULL);

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_66_row_first_slice_pset */
\pset format csv

SET pgx_lower.execution_mode = 'auto';
SET pgx_lower.route_path_notices = on;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_66_row_first_slice_001 */
SELECT id FROM row_first_slice WHERE id = 1;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_66_row_first_slice_002 */
SELECT id, payload FROM row_first_slice WHERE payload = 20;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_66_row_first_slice_003 */
SELECT id FROM row_first_slice WHERE payload = payload AND id = 13;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=legacy id=pgx_66_row_first_slice_004 */
SELECT id FROM row_first_slice ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=legacy id=pgx_66_row_first_slice_005 */
SELECT id FROM row_first_slice WHERE id <> 7;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_66_row_first_slice_006 */
SELECT row_number() OVER () FROM row_first_slice;

SET pgx_lower.route_path_notices = off;

DROP TABLE row_first_slice;
