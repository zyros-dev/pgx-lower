LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_nulls;

CREATE TABLE test_nulls
(
    id             SERIAL PRIMARY KEY,
    nullable_value INTEGER,
    backup_value   INTEGER,
    third_value    INTEGER
);

INSERT INTO test_nulls(nullable_value, backup_value, third_value)
VALUES (NULL, 100, 200),
       (10, 110, 210),
       (NULL, 120, 220),
       (30, NULL, 230),
       (40, 140, NULL);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_12_null_handling_001 */
SELECT (nullable_value IS NULL) AS is_null_check
FROM test_nulls;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_12_null_handling_002 */
SELECT (nullable_value IS NOT NULL) AS is_not_null_check
FROM test_nulls;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_12_null_handling_003 */
SELECT (backup_value IS NULL) AS backup_is_null
FROM test_nulls;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_12_null_handling_004 */
SELECT (backup_value IS NOT NULL) AS backup_is_not_null
FROM test_nulls;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_12_null_handling_005 */
SELECT COALESCE(nullable_value, backup_value) AS coalesced_value
FROM test_nulls;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_12_null_handling_006 */
SELECT COALESCE(nullable_value, backup_value, third_value) AS triple_coalesce
FROM test_nulls;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_12_null_handling_007 */
SELECT COALESCE(nullable_value, -1) AS coalesce_with_constant
FROM test_nulls;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_12_null_handling_008 */
SELECT COALESCE(10, 20) AS coalesce_non_nullable_constants;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_12_null_handling_009 */
SELECT COALESCE(100, 200, 300) AS coalesce_triple_non_nullable;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_12_null_handling_010 */
SELECT COALESCE(id, 999) AS coalesce_non_null_column
FROM test_nulls;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_12_null_handling_011 */
SELECT COALESCE(id, id + 1000) AS coalesce_non_null_expr
FROM test_nulls;

DROP TABLE test_nulls;