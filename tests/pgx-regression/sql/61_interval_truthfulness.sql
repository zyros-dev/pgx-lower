LOAD 'pgx_lower.so';
SET DateStyle = 'ISO, MDY';
SET IntervalStyle = 'postgres';
SET TimeZone = 'UTC';
SET extra_float_digits = 0;

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_61_interval_truthfulness_pset */
\pset format csv

CREATE TEMP TABLE interval_day_truthfulness(
    id int4,
    d date
);

CREATE TEMP TABLE interval_month_truthfulness(
    id int4,
    d date,
    i interval
);

CREATE TEMP TABLE interval_timestamp_truthfulness(
    id int4,
    start_ts timestamp,
    end_ts timestamp
);

CREATE TEMP TABLE interval_sort_truthfulness(
    id int4,
    i interval
);

INSERT INTO interval_day_truthfulness VALUES
    (1, date '1998-12-01');

INSERT INTO interval_month_truthfulness VALUES
    (2, date '2000-01-31', interval '1 month');

INSERT INTO interval_timestamp_truthfulness VALUES
    (1, timestamp '2000-01-01 00:00:00', timestamp '2000-01-02 00:00:00');

INSERT INTO interval_sort_truthfulness VALUES
    (2, interval '1 month'),
    (1, interval '5 days'),
    (3, NULL);

SET pgx_lower.execution_mode = 'auto';

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_61_interval_truthfulness_001 */
SELECT d - interval '90 days' FROM interval_day_truthfulness WHERE id = 1;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_002 */
SELECT d + i FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_61_interval_truthfulness_003 */
SELECT i FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_004 */
SELECT i = interval '1 month' FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_005 */
SELECT i < interval '2 months' FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_006 */
SELECT timestamp '2000-01-01 00:00:00' + i FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_007 */
SELECT end_ts - start_ts FROM interval_timestamp_truthfulness WHERE id = 1;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_008 */
SELECT i + interval '1 day' FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_009 */
SELECT i - interval '1 day' FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_010 */
SELECT i + d FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_011 */
SELECT i + timestamp '2000-01-01 00:00:00' FROM interval_month_truthfulness WHERE id = 2;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_61_interval_truthfulness_012 */
SELECT sum(i), avg(i), min(i), max(i) FROM interval_month_truthfulness;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_61_interval_truthfulness_013 */
SELECT id, i FROM interval_sort_truthfulness ORDER BY id;

DROP TABLE interval_day_truthfulness;
DROP TABLE interval_month_truthfulness;
DROP TABLE interval_timestamp_truthfulness;
DROP TABLE interval_sort_truthfulness;
