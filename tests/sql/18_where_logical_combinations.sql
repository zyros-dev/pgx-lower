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

SELECT id, age, score
FROM test_where_logical
WHERE age > 25
  AND score > 85;
SELECT department, age
FROM test_where_logical
WHERE active = true
  AND age < 30;
SELECT id, department
FROM test_where_logical
WHERE score >= 85
  AND department = 'Engineering';
SELECT id, age, score
FROM test_where_logical
WHERE age < 25
   OR score > 90;
SELECT department, active
FROM test_where_logical
WHERE department = 'Sales'
   OR department = 'Marketing';
SELECT id, age
FROM test_where_logical
WHERE age > 35
   OR active = false;
SELECT id, age, department
FROM test_where_logical
WHERE NOT active;
SELECT age, score
FROM test_where_logical
WHERE NOT (age < 25);

SELECT department
FROM test_where_logical
WHERE NOT (department = 'Engineering');

SELECT id, age, score
FROM test_where_logical
WHERE (age > 25 AND score > 80)
   OR (active = false);

SELECT department, age
FROM test_where_logical
WHERE active = true
  AND (age < 30 OR score > 85);

SELECT id, department
FROM test_where_logical
WHERE NOT (age < 25 OR score < 80);

SELECT age, score, department
FROM test_where_logical
WHERE age > 25
  AND score > 80
  AND active = true;
SELECT id, age
FROM test_where_logical
WHERE age < 30
   OR score > 90
   OR department = 'Sales';
DROP TABLE test_where_logical;
