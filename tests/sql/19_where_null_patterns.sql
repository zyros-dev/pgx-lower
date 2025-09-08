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

SELECT id, name, age
FROM test_where_nulls
WHERE age IS NULL;
SELECT id, name, email
FROM test_where_nulls
WHERE email IS NULL;
SELECT id, name, score
FROM test_where_nulls
WHERE score IS NULL;
SELECT id, name, age
FROM test_where_nulls
WHERE age IS NOT NULL;
SELECT id, name, email
FROM test_where_nulls
WHERE email IS NOT NULL;
SELECT id, name, score
FROM test_where_nulls
WHERE score IS NOT NULL;
SELECT id, name
FROM test_where_nulls
WHERE age = NULL;
SELECT id, name
FROM test_where_nulls
WHERE email <> NULL;
SELECT id, name, age
FROM test_where_nulls
WHERE age IS NULL
   OR age < 30;
SELECT id, name, email
FROM test_where_nulls
WHERE email IS NOT NULL
  AND age > 25;
SELECT id, name
FROM test_where_nulls
WHERE score IS NULL
  AND age IS NOT NULL;
SELECT id, name, age
FROM test_where_nulls
WHERE COALESCE(age, 0) > 25;
SELECT id, name, score
FROM test_where_nulls
WHERE COALESCE(score, 0) >= 85;
SELECT id, name
FROM test_where_nulls
WHERE age IS NULL
  AND email IS NULL;
SELECT id, name
FROM test_where_nulls
WHERE age IS NOT NULL
  AND email IS NOT NULL
  AND score IS NOT NULL;
SELECT id, name
FROM test_where_nulls
WHERE age IS NULL
   OR email IS NULL
   OR score IS NULL;
SELECT id, name, age
FROM test_where_nulls
WHERE (age IS NULL)
   OR (age > 30 AND score IS NOT NULL);

SELECT id, name
FROM test_where_nulls
WHERE NOT (age IS NULL);

DROP TABLE test_where_nulls;
