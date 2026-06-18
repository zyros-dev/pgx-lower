LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS numeric_correctness;
DROP TABLE IF EXISTS numeric_untyped;

CREATE TABLE numeric_correctness
(
    label TEXT,
    n     NUMERIC
);

INSERT INTO numeric_correctness(label, n)
VALUES ('pos', 12345.6789),
       ('neg', -12345.6789),
       ('boundary38', 99999999999999999999999999999999999999),
       ('wide40', 1000000000000000000000000000000000000000),
       ('verywide100', '9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999'::NUMERIC),
       ('scale18', 1.000000000000000001),
       ('nan', 'NaN'::NUMERIC),
       ('inf', 'Infinity'::NUMERIC),
       ('ninf', '-Infinity'::NUMERIC);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_44_numeric_correctness_001 */
SELECT label, n
FROM numeric_correctness
ORDER BY n, label;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_44_numeric_correctness_002 */
SELECT label,
       n = 'NaN'::NUMERIC AS eq_nan,
       n > 99999999999999999999999999999999999999::NUMERIC AS gt_boundary,
       n < 0::NUMERIC AS lt_zero
FROM numeric_correctness
ORDER BY label;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_44_numeric_correctness_003 */
SELECT SUM(n) AS finite_sum,
       MIN(n) AS finite_min,
       MAX(n) AS finite_max
FROM numeric_correctness
WHERE label IN ('pos', 'neg', 'scale18', 'boundary38');

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_44_numeric_correctness_004 */
SELECT 1000000000000000000000000000000000000000::NUMERIC + 1::NUMERIC AS wide_literal_plus_one;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_44_numeric_correctness_005 */
SELECT '9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999'::NUMERIC * 2::NUMERIC AS verywide_product;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_44_numeric_correctness_006 */
SELECT 'NaN'::NUMERIC = 'NaN'::NUMERIC AS nan_eq_nan,
       'NaN'::NUMERIC > 'Infinity'::NUMERIC AS nan_gt_inf,
       'Infinity'::NUMERIC > 99999999999999999999999999999999999999::NUMERIC AS inf_gt_wide,
       '-Infinity'::NUMERIC < -99999999999999999999999999999999999999::NUMERIC AS ninf_lt_wide;

CREATE TABLE numeric_untyped
(
    id INT,
    n  NUMERIC
);

INSERT INTO numeric_untyped(id, n)
VALUES (1, 1.2),
       (2, 1.234567890123456789),
       (3, 123456789012345678901234567890123456789.987654321),
       (4, -0.000000000000000001);

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_44_numeric_correctness_007 */
SELECT id,
       n,
       n + 0.000000000000000001 AS plus_1e_minus_18,
       n * 3 AS triple,
       n::TEXT AS as_text
FROM numeric_untyped
ORDER BY id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_44_numeric_correctness_008 */
SELECT CAST('1.000000000000000001' AS NUMERIC) + CAST('2' AS NUMERIC) AS string_numeric_cast;

DROP TABLE numeric_untyped;
DROP TABLE numeric_correctness;
