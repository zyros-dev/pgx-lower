LOAD
'pgx_lower';

DROP TABLE IF EXISTS test_comparison;

CREATE TABLE test_comparison
(
    id    SERIAL PRIMARY KEY,
    value INTEGER,
    score INTEGER
);

INSERT INTO test_comparison(value, score)
VALUES (10, 15),
       (20, 20),
       (15, 10),
       (25, 5),
       (30, 30);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_10_comparison_ops_001 */
SELECT (value = score) AS is_equal
FROM test_comparison;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_10_comparison_ops_002 */
SELECT (value <> score) AS not_equal
FROM test_comparison;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_10_comparison_ops_003 */
SELECT (value != score) AS not_equal_alt
FROM test_comparison;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_10_comparison_ops_004 */
SELECT (value < score) AS less_than
FROM test_comparison;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_10_comparison_ops_005 */
SELECT (value <= score) AS less_equal
FROM test_comparison;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_10_comparison_ops_006 */
SELECT (value > score) AS greater_than
FROM test_comparison;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_10_comparison_ops_007 */
SELECT (value >= score) AS greater_equal
FROM test_comparison;

DROP TABLE test_comparison;