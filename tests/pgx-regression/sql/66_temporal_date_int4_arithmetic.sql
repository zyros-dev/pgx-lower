LOAD 'pgx_lower.so';
SET DateStyle = 'ISO, MDY';
SET IntervalStyle = 'postgres';
SET TimeZone = 'UTC';
SET extra_float_digits = 0;

CREATE TEMP TABLE temporal_date_int4_arithmetic(
    id int4,
    d date,
    n int4
);

INSERT INTO temporal_date_int4_arithmetic VALUES
    (1, date '2000-01-01', 7),
    (2, date '1999-12-31', -1);

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_66_temporal_date_int4_arithmetic_pset */
\pset format csv

SET pgx_lower.execution_mode = 'auto';

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_66_temporal_date_int4_arithmetic_001 */
SELECT id, d + 7 AS plus_const FROM temporal_date_int4_arithmetic ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_66_temporal_date_int4_arithmetic_002 */
SELECT id, d - 7 AS minus_const FROM temporal_date_int4_arithmetic ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_66_temporal_date_int4_arithmetic_003 */
SELECT id, d + n AS plus_column FROM temporal_date_int4_arithmetic ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_66_temporal_date_int4_arithmetic_004 */
SELECT id, d - n AS minus_column FROM temporal_date_int4_arithmetic ORDER BY id;

DROP TABLE temporal_date_int4_arithmetic;
