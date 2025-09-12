LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_order_where;

CREATE TABLE test_order_where
(
    id         SERIAL PRIMARY KEY,
    department VARCHAR(20),
    salary     INTEGER,
    years      INTEGER,
    age        INTEGER,
    name       VARCHAR(50)
);

INSERT INTO test_order_where(department, salary, years, age, name)
VALUES ('Sales', 50000, 3, 28, 'Alice'),
       ('IT', 60000, 2, 25, 'Bob'),
       ('Sales', 45000, 5, 32, 'Carol'),
       ('IT', 65000, 4, 30, 'David'),
       ('Sales', 50000, 1, 24, 'Eve'),
       ('IT', 60000, 6, 35, 'Frank'),
       ('HR', 55000, 3, 29, 'Grace'),
       ('HR', 48000, 2, 26, 'Henry');

SELECT name, salary
FROM test_order_where
WHERE salary > 50000
ORDER BY salary;
SELECT name, department, salary
FROM test_order_where
WHERE salary >= 50000
ORDER BY name;
SELECT name, department, years
FROM test_order_where
WHERE years > 2
  AND department = 'IT'
ORDER BY years;
SELECT name, age, salary
FROM test_order_where
WHERE age < 30
ORDER BY salary DESC;
SELECT name, department, salary, years
FROM test_order_where
WHERE salary >= 45000
ORDER BY department, salary DESC;

SELECT name, department, age
FROM test_order_where
WHERE department IN ('IT', 'Sales')
ORDER BY department, age;
SELECT name, salary, age
FROM test_order_where
WHERE salary <> 50000
ORDER BY age, salary;
DROP TABLE test_order_where;
