LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS test_distinct;

CREATE TABLE test_distinct
(
    id       SERIAL PRIMARY KEY,
    category INTEGER,
    status   INTEGER,
    value    INTEGER,
    name     TEXT
);

INSERT INTO test_distinct(category, status, value, name)
VALUES (1, 10, 100, 'apple'),
       (2, 20, 200, 'banana'),
       (1, 10, 150, 'apple'),
       (3, 30, 300, 'cherry'),
       (2, 20, 250, 'banana'),
       (1, 15, 100, 'apricot'),
       (3, 30, 300, 'cherry'),
       (4, 40, 400, 'date'),
       (2, 25, 200, 'blueberry'),
       (1, 10, 100, 'apple');


SELECT DISTINCT category
FROM test_distinct;


SELECT DISTINCT category
FROM test_distinct
ORDER BY category;


SELECT DISTINCT category, status
FROM test_distinct
ORDER BY category, status;

SET client_min_messages TO DEBUG1;
SET pgx_lower.log_enable = true;
SET pgx_lower.log_debug = true;
SET pgx_lower.log_io = true;
SET pgx_lower.log_ir = true;
SET pgx_lower.log_trace = true;
SET pgx_lower.log_verbose = true;
SET pgx_lower.enabled_categories = 'AST_TRANSLATE,RELALG_LOWER,DB_LOWER,DSA_LOWER,UTIL_LOWER,RUNTIME,JIT,GENERAL';

SELECT DISTINCT category + status AS combined
FROM test_distinct
ORDER BY combined;

SELECT DISTINCT name
FROM test_distinct
ORDER BY name;

SELECT DISTINCT category
FROM test_distinct
WHERE value >= 200
ORDER BY category;

SELECT DISTINCT category, value
FROM test_distinct
WHERE status <= 20
ORDER BY category, value;

SELECT DISTINCT value / 100 AS hundreds
FROM test_distinct
ORDER BY hundreds;

SELECT ALL category
FROM test_distinct
ORDER BY category;

SELECT SUM(category)
FROM test_distinct
ORDER BY category;

SELECT SUM(DISTINCT category)
FROM test_distinct
ORDER BY category;

DROP TABLE test_distinct;
