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

SELECT name, age
FROM test_where_simple
WHERE age = 25;
SELECT name, score
FROM test_where_simple
WHERE score = 92;
SELECT id, name
FROM test_where_simple
WHERE name = 3;

SELECT name, age
FROM test_where_simple
WHERE age <> 25;
SELECT name, score
FROM test_where_simple
WHERE score != 88;

SELECT name, age
FROM test_where_simple
WHERE age > 25;
SELECT name, age
FROM test_where_simple
WHERE age >= 30;
SELECT name, score
FROM test_where_simple
WHERE score < 90;
SELECT name, score
FROM test_where_simple
WHERE score <= 85;

SELECT name
FROM test_where_simple
WHERE age > 20;
SELECT name
FROM test_where_simple
WHERE score >= 85;

DROP TABLE test_where_simple;