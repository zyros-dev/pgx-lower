LOAD 'pgx_lower.so';

CREATE TEMP TABLE string_boundary_supported(
    id int4,
    t text,
    v varchar(8),
    c char(5)
);

INSERT INTO string_boundary_supported VALUES
    (1, 'alpha', 'alpha', 'A'),
    (2, 'beta', 'beta', 'B');

CREATE TEMP TABLE string_boundary_unsupported(
    id int4,
    b bytea
);

INSERT INTO string_boundary_unsupported VALUES
    (1, decode('616c706861', 'hex')),
    (2, decode('62657461', 'hex'));

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_62_string_support_boundary_pset */
\pset format csv

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_62_string_support_boundary_001 */
SELECT id, t, v, c, id AS witness FROM string_boundary_supported ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_62_string_support_boundary_002 */
SELECT id,
       'const-text'::text AS t,
       'const-v'::varchar(8) AS v,
       'C'::char(5) AS c,
       id AS witness
FROM string_boundary_supported
WHERE id = 1;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_003 */
SELECT b, id AS witness FROM string_boundary_unsupported WHERE id = 1;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_004 */
SELECT 'x'::"char" AS char_value, id AS witness FROM string_boundary_supported WHERE id = 1;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_005 */
SELECT 'pgx_lower'::name AS name_value, id AS witness FROM string_boundary_supported WHERE id = 1;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_006 */
SELECT 'pgx'::cstring AS cstring_value, id AS witness FROM string_boundary_supported WHERE id = 1;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_007 */
SELECT t, id AS witness FROM string_boundary_supported WHERE t = 'alpha'::text;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_008 */
SELECT c, id AS witness FROM string_boundary_supported WHERE c = 'A'::char(5);

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_009 */
SELECT t, id AS witness FROM string_boundary_supported WHERE t LIKE 'a%'::text;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_010 */
SELECT c, id AS witness FROM string_boundary_supported WHERE c LIKE 'A%'::text;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_62_string_support_boundary_011 */
SELECT v::text AS cast_value, id AS witness FROM string_boundary_supported WHERE id = 1;

DROP TABLE string_boundary_unsupported;
DROP TABLE string_boundary_supported;
