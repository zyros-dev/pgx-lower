LOAD 'pgx_lower.so';
SET DateStyle = 'ISO, MDY';
SET IntervalStyle = 'postgres';
SET TimeZone = 'UTC';
SET extra_float_digits = 0;

CREATE TEMP TABLE temporal_date_extract(
    id int4,
    d date
);

INSERT INTO temporal_date_extract VALUES
    (1, date '2000-01-01'),
    (2, date '1999-12-31'),
    (3, date '2000-01-31'),
    (4, NULL);

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_67_temporal_date_extract_pset */
\pset format csv

SET pgx_lower.execution_mode = 'auto';

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_67_temporal_date_extract_auto */
SELECT id, d, extract(year from d) AS extract_year, extract(month from d) AS extract_month,
       extract(day from d) AS extract_day
FROM temporal_date_extract
ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_67_temporal_date_extract_literal */
SELECT id, extract(year from date '2000-01-31') AS extract_year,
       extract(month from date '2000-01-31') AS extract_month,
       extract(day from date '2000-01-31') AS extract_day
FROM temporal_date_extract
WHERE id = 1;

SET pgx_lower.execution_mode = 'force_lower';

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_67_temporal_date_extract_force */
SELECT id, d, extract(year from d) AS extract_year, extract(month from d) AS extract_month,
       extract(day from d) AS extract_day
FROM temporal_date_extract
ORDER BY id;

DROP TABLE temporal_date_extract;
