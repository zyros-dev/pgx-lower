LOAD 'pgx_lower.so';

CREATE TEMP TABLE string_scalar_truthfulness(
    id int4,
    t text,
    v varchar(20),
    c3 char(3),
    c5 char(5)
);

INSERT INTO string_scalar_truthfulness VALUES
    (1, 'alpha', 'alpha', 'ab', 'ab'),
    (2, 'beta', 'beta', 'xy', 'xy'),
    (3, convert_from(decode('c3a9636c616972', 'hex'), 'UTF8'),
        convert_from(decode('c3a9636c616972', 'hex'), 'UTF8')::varchar(20), 'zz', 'zz');

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_64_string_scalar_truthfulness_pset */
\pset format csv

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_64_string_scalar_truthfulness_001 */
SELECT id, t = v::text AS text_varchar_eq FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_64_string_scalar_truthfulness_002 */
SELECT id, c3 = c5 AS bpchar_cross_width_eq FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_64_string_scalar_truthfulness_003 */
SELECT id, t LIKE 'a%' AS starts_with_a, t NOT LIKE 'b%' AS not_b FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_64_string_scalar_truthfulness_004 */
SELECT id, v LIKE 'a%' AS varchar_starts_with_a, c5 LIKE 'a%' AS bpchar_starts_with_a
FROM string_scalar_truthfulness
ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_64_string_scalar_truthfulness_005 */
SELECT id, substring(t from 1 for 1) AS text_first_char, substring(v from 1 for 1) AS varchar_first_char
FROM string_scalar_truthfulness
ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_64_string_scalar_truthfulness_006 */
SELECT id, upper(t) AS upper_t, lower(t) AS lower_t FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_64_string_scalar_truthfulness_007 */
SELECT id, upper(v) AS upper_v, lower(v) AS lower_v FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_64_string_scalar_truthfulness_008 */
SELECT t COLLATE "C" < 'z' COLLATE "C" FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_64_string_scalar_truthfulness_009 */
SELECT t || v FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_64_string_scalar_truthfulness_010 */
SELECT length(t) FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_64_string_scalar_truthfulness_011 */
SELECT id, substring(c5 from 1 for 1) AS bpchar_first_char FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_64_string_scalar_truthfulness_012 */
SELECT id, upper(c5) AS upper_c5, lower(c5) AS lower_c5 FROM string_scalar_truthfulness ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_64_string_scalar_truthfulness_013 */
SELECT id,
       t < 'delta' AS text_lt_delta,
       v < 'delta'::varchar(20) AS varchar_lt_delta,
       c5 < 'zzzzz'::char(5) AS bpchar_lt_zzzzz
FROM string_scalar_truthfulness
ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_64_string_scalar_truthfulness_014 */
SELECT id, t::varchar(8) AS text_to_varchar FROM string_scalar_truthfulness ORDER BY id;

DROP TABLE string_scalar_truthfulness;
