LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_where_logical;

CREATE TABLE test_where_logical
(
    id         SERIAL PRIMARY KEY,
    age        INTEGER,
    score      INTEGER,
    active     BOOLEAN,
    department VARCHAR(20)
);

INSERT INTO test_where_logical(age, score, active, department)
VALUES (25, 85, true, 'Engineering'),
       (30, 92, false, 'Marketing'),
       (22, 78, true, 'Engineering'),
       (35, 88, true, 'Sales'),
       (28, 95, false, 'Marketing'),
       (32, 72, true, 'Sales'),
       (26, 89, false, 'Engineering');

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_001 */
SELECT id, age, score
FROM test_where_logical
WHERE age > 25
  AND score > 85;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_002 */
SELECT department, age
FROM test_where_logical
WHERE active = true
  AND age < 30;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_18_where_logical_combinations_003 */
SELECT id, department
FROM test_where_logical
WHERE score >= 85
  AND department = 'Engineering';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_004 */
SELECT id, age, score
FROM test_where_logical
WHERE age < 25
   OR score > 90;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_18_where_logical_combinations_005 */
SELECT department, active
FROM test_where_logical
WHERE department = 'Sales'
   OR department = 'Marketing';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_006 */
SELECT id, age
FROM test_where_logical
WHERE age > 35
   OR active = false;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_007 */
SELECT id, age, department
FROM test_where_logical
WHERE NOT active;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_008 */
SELECT age, score
FROM test_where_logical
WHERE NOT (age < 25);

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_18_where_logical_combinations_009 */
SELECT department
FROM test_where_logical
WHERE NOT (department = 'Engineering');

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_010 */
SELECT id, age, score
FROM test_where_logical
WHERE (age > 25 AND score > 80)
   OR (active = false);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_011 */
SELECT department, age
FROM test_where_logical
WHERE active = true
  AND (age < 30 OR score > 85);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_012 */
SELECT id, department
FROM test_where_logical
WHERE NOT (age < 25 OR score < 80);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_18_where_logical_combinations_013 */
SELECT age, score, department
FROM test_where_logical
WHERE age > 25
  AND score > 80
  AND active = true;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_18_where_logical_combinations_014 */
SELECT id, age
FROM test_where_logical
WHERE age < 30
   OR score > 90
   OR department = 'Sales';
DROP TABLE test_where_logical;
