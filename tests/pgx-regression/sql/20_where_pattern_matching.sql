LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_where_patterns;

CREATE TABLE test_where_patterns
(
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(50),
    email      VARCHAR(100),
    department VARCHAR(20),
    salary     INTEGER,
    status     VARCHAR(10)
);

INSERT INTO test_where_patterns(name, email, department, salary, status)
VALUES ('Alice Smith', 'alice.smith@company.com', 'Engineering', 75000, 'active'),
       ('Bob Johnson', 'bob.j@company.com', 'Marketing', 65000, 'inactive'),
       ('Carol Davis', 'carol.davis@company.com', 'Engineering', 80000, 'active'),
       ('David Wilson', 'david@company.com', 'Sales', 70000, 'active'),
       ('Eve Brown', 'eve.brown@company.com', 'Marketing', 68000, 'pending'),
       ('Frank Miller', 'frank.m@company.com', 'Sales', 72000, 'active'),
       ('Grace Taylor', 'grace@company.com', 'Engineering', 85000, 'inactive');

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_001 */
SELECT id, name
FROM test_where_patterns
WHERE name LIKE 'A%';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_002 */
SELECT id, name
FROM test_where_patterns
WHERE name LIKE '%Smith';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_003 */
SELECT id, name
FROM test_where_patterns
WHERE name LIKE '%o%';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_004 */
SELECT id, email
FROM test_where_patterns
WHERE email LIKE '%.%@%';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_005 */
SELECT id, email
FROM test_where_patterns
WHERE email LIKE '%company.com';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_006 */
SELECT id, name
FROM test_where_patterns
WHERE name LIKE '___ %';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_007 */
SELECT id, name
FROM test_where_patterns
WHERE name LIKE '% _____';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_008 */
SELECT department
FROM test_where_patterns
WHERE department LIKE '%ing';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_009 */
SELECT id, name
FROM test_where_patterns
WHERE name NOT LIKE 'A%';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_010 */
SELECT id, email
FROM test_where_patterns
WHERE email NOT LIKE '%@company.com';
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_20_where_pattern_matching_011 */
SELECT id, name, department
FROM test_where_patterns
WHERE department IN ('Engineering', 'Sales');

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_20_where_pattern_matching_012 */
SELECT id, name, status
FROM test_where_patterns
WHERE status IN ('active', 'pending');

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_013 */
SELECT id, name
FROM test_where_patterns
WHERE salary IN (75000, 80000, 85000);

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_20_where_pattern_matching_014 */
SELECT id, name, department
FROM test_where_patterns
WHERE department NOT IN ('Marketing');

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_20_where_pattern_matching_015 */
SELECT id, name, status
FROM test_where_patterns
WHERE status NOT IN ('inactive');

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_20_where_pattern_matching_016 */
SELECT id, name, department
FROM test_where_patterns
WHERE name LIKE 'C%'
  AND department = 'Engineering';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_20_where_pattern_matching_017 */
SELECT id, name, salary
FROM test_where_patterns
WHERE (name LIKE '%e%' OR name LIKE '%a%')
  AND salary > 70000;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_20_where_pattern_matching_018 */
SELECT id, name
FROM test_where_patterns
WHERE department IN ('Engineering', 'Sales')
  AND status = 'active';
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_20_where_pattern_matching_019 */
SELECT id, name, email
FROM test_where_patterns
WHERE name LIKE '%a%'
  AND email LIKE '%@company.com'
  AND department NOT IN ('Marketing');

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_20_where_pattern_matching_020 */
SELECT id, name, department
FROM test_where_patterns
WHERE (name NOT LIKE 'A%' AND name NOT LIKE 'B%')
   OR department = 'Engineering';
DROP TABLE test_where_patterns;
