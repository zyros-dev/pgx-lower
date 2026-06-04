LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_where_simple;

CREATE TABLE test_where_simple
(
    id    SERIAL PRIMARY KEY,
    age   INTEGER,
    score INTEGER,
    name  INTEGER
);

INSERT INTO test_where_simple(age, score, name)
VALUES (25, 85, 1),
       (30, 92, 2),
       (22, 78, 3),
       (35, 88, 4),
       (28, 95, 5);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_001 */
SELECT name, age
FROM test_where_simple
WHERE age = 25;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_002 */
SELECT name, score
FROM test_where_simple
WHERE score = 92;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_003 */
SELECT id, name
FROM test_where_simple
WHERE name = 3;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_004 */
SELECT name, age
FROM test_where_simple
WHERE age <> 25;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_005 */
SELECT name, score
FROM test_where_simple
WHERE score != 88;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_006 */
SELECT name, age
FROM test_where_simple
WHERE age > 25;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_007 */
SELECT name, age
FROM test_where_simple
WHERE age >= 30;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_008 */
SELECT name, score
FROM test_where_simple
WHERE score < 90;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_009 */
SELECT name, score
FROM test_where_simple
WHERE score <= 85;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_010 */
SELECT name
FROM test_where_simple
WHERE age > 20;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_17_where_simple_conditions_011 */
SELECT name
FROM test_where_simple
WHERE score >= 85;

DROP TABLE test_where_simple;