LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_order_multi;

CREATE TABLE test_order_multi
(
    id         SERIAL PRIMARY KEY,
    department VARCHAR(20),
    salary     INTEGER,
    years      INTEGER,
    name       VARCHAR(50)
);

INSERT INTO test_order_multi(department, salary, years, name)
VALUES ('Sales', 50000, 3, 'Alice'),
       ('IT', 60000, 2, 'Bob'),
       ('Sales', 45000, 5, 'Carol'),
       ('IT', 65000, 4, 'David'),
       ('Sales', 50000, 1, 'Eve'),
       ('IT', 60000, 6, 'Frank'),
       ('HR', 55000, 3, 'Grace');

SELECT department, salary, name
FROM test_order_multi
ORDER BY department, salary;
SELECT department, salary, name
FROM test_order_multi
ORDER BY department ASC, salary ASC;
SELECT department, salary, name
FROM test_order_multi
ORDER BY department ASC, salary DESC;
SELECT department, salary, name
FROM test_order_multi
ORDER BY department DESC, salary DESC;
SELECT department, salary, years, name
FROM test_order_multi
ORDER BY department, salary, years;
SELECT department, salary, years, name
FROM test_order_multi
ORDER BY department ASC, salary DESC, years ASC;
DROP TABLE test_order_multi;
