LOAD 'pgx_lower.so';

CREATE TEMP TABLE row_first_slice(
    id int8 NOT NULL,
    payload int4,
    note text
);

INSERT INTO row_first_slice VALUES
    (1, 10, 'alpha'),
    (2, NULL, 'bravo'),
    (3, 30, 'charlie');

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_66_row_first_slice_pset */
\pset format csv

SET pgx_lower.execution_mode = 'auto';
SET pgx_lower.route_path_notices = on;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_66_row_first_slice_row */
SELECT id, payload FROM row_first_slice WHERE id < 3;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=legacy id=pgx_66_row_first_slice_legacy */
SELECT id + 1 AS id_plus_one FROM row_first_slice;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_66_row_first_slice_fallback */
SELECT generate_series(1, 2);

SET pgx_lower.route_path_notices = off;

DROP TABLE row_first_slice;
