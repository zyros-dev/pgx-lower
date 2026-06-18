LOAD 'pgx_lower.so';

CREATE TEMP TABLE row_primitive_surface(
    flag bool,
    small int2,
    i4 int4,
    i8 int8 NOT NULL,
    f4 float4,
    f8 float8,
    amount numeric(12,2),
    d date,
    ts timestamp,
    iv interval,
    t text,
    v varchar(16),
    c char(4)
);

CREATE TEMP TABLE row_primitive_unsupported(
    i8 int8 NOT NULL,
    b bytea,
    t text,
    iv_ym interval year to month
);

CREATE TEMP TABLE row_primitive_collated(
    i8 int8 NOT NULL,
    t text COLLATE "C"
);

INSERT INTO row_primitive_surface VALUES
    (true, 1, 1, 100, 1.25, 10.5, 12.34, '1998-12-01', '1998-12-01 08:30:00',
     '1 day', 'alpha', 'alpha', 'ALFA'),
    (false, 2, 2, 200, 2.5, 20.75, 56.78, '1998-12-02', '1998-12-02 09:45:30',
     '2 days 03:04:05', 'beta', 'beta', 'BETA'),
    (NULL, NULL, NULL, 300, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

INSERT INTO row_primitive_unsupported VALUES
    (100, decode('61', 'hex'), 'alpha', '1 year');

INSERT INTO row_primitive_collated VALUES
    (100, 'alpha');

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_68_row_primitive_surface_pset */
\pset format csv

SET DateStyle = 'ISO, YMD';
SET IntervalStyle = 'postgres';
SET pgx_lower.execution_mode = 'auto';
SET pgx_lower.route_path_notices = on;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_68_row_primitive_surface_001 */
SELECT flag, small, i4, i8, f4, f8, amount, d, ts, iv
FROM row_primitive_surface
WHERE i8 = 100;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_68_row_primitive_surface_002 */
SELECT t, v, c
FROM row_primitive_surface
WHERE i8 = 200;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_68_row_primitive_surface_003 */
SELECT flag IS NULL AS flag_null,
       small IS NULL AS small_null,
       i4 IS NULL AS i4_null,
       amount IS NULL AS amount_null,
       d IS NULL AS d_null,
       ts IS NULL AS ts_null,
       iv IS NULL AS iv_null,
       t IS NULL AS t_null
FROM row_primitive_surface
WHERE i8 = 300;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=row id=pgx_68_row_primitive_surface_004 */
SELECT i8
FROM row_primitive_surface
WHERE flag;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=legacy id=pgx_68_row_primitive_surface_005 */
SELECT i8, count(*)
FROM row_primitive_surface
GROUP BY i8
HAVING i8 = 100;

/* <<pgx-lower-config>>: auto_should_route_to=lower lower_path=legacy id=pgx_68_row_primitive_surface_006 */
SELECT t
FROM row_primitive_surface
ORDER BY i8;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_68_row_primitive_surface_007 */
SELECT b
FROM row_primitive_unsupported
WHERE i8 = 100;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_68_row_primitive_surface_008 */
SELECT t
FROM row_primitive_collated
WHERE i8 = 100;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_68_row_primitive_surface_009 */
SELECT iv_ym
FROM row_primitive_unsupported
WHERE i8 = 100;

SET pgx_lower.route_path_notices = off;

DROP TABLE row_primitive_surface;
DROP TABLE row_primitive_unsupported;
DROP TABLE row_primitive_collated;
