LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_where_nulls;

CREATE TABLE test_where_nulls
(
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(50),
    age   INTEGER,
    email VARCHAR(100),
    score INTEGER
);

INSERT INTO test_where_nulls(name, age, email, score)
VALUES ('Alice', 25, 'alice@test.com', 85),
       ('Bob', NULL, 'bob@test.com', 92),
       ('Carol', 30, NULL, 78),
       ('David', 35, 'david@test.com', NULL),
       ('Eve', NULL, NULL, 95),
       ('Frank', 28, 'frank@test.com', NULL),
       ('Grace', 32, NULL, 88);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_001 */
SELECT id, name, age
FROM test_where_nulls
WHERE age IS NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_002 */
SELECT id, name, email
FROM test_where_nulls
WHERE email IS NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_003 */
SELECT id, name, score
FROM test_where_nulls
WHERE score IS NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_004 */
SELECT id, name, age
FROM test_where_nulls
WHERE age IS NOT NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_005 */
SELECT id, name, email
FROM test_where_nulls
WHERE email IS NOT NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_006 */
SELECT id, name, score
FROM test_where_nulls
WHERE score IS NOT NULL;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_19_where_null_patterns_007 */
SELECT id, name
FROM test_where_nulls
WHERE age = NULL;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_19_where_null_patterns_008 */
SELECT id, name
FROM test_where_nulls
WHERE email <> NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_009 */
SELECT id, name, age
FROM test_where_nulls
WHERE age IS NULL
   OR age < 30;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_010 */
SELECT id, name, email
FROM test_where_nulls
WHERE email IS NOT NULL
  AND age > 25;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_011 */
SELECT id, name
FROM test_where_nulls
WHERE score IS NULL
  AND age IS NOT NULL;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_19_where_null_patterns_012 */
SELECT id, name, age
FROM test_where_nulls
WHERE COALESCE(age, 0) > 25;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_19_where_null_patterns_013 */
SELECT id, name, score
FROM test_where_nulls
WHERE COALESCE(score, 0) >= 85;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_014 */
SELECT id, name
FROM test_where_nulls
WHERE age IS NULL
  AND email IS NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_015 */
SELECT id, name
FROM test_where_nulls
WHERE age IS NOT NULL
  AND email IS NOT NULL
  AND score IS NOT NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_016 */
SELECT id, name
FROM test_where_nulls
WHERE age IS NULL
   OR email IS NULL
   OR score IS NULL;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_017 */
SELECT id, name, age
FROM test_where_nulls
WHERE (age IS NULL)
   OR (age > 30 AND score IS NOT NULL);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_19_where_null_patterns_018 */
SELECT id, name
FROM test_where_nulls
WHERE NOT (age IS NULL);

DROP TABLE test_where_nulls;
