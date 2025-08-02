-- Test ORDER BY combined with WHERE clauses: filtering and sorting together
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → SubOp → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_order_where;

CREATE TABLE test_order_where (
    id SERIAL PRIMARY KEY,
    department VARCHAR(20),
    salary INTEGER,
    years INTEGER,
    age INTEGER,
    name VARCHAR(50)
);

INSERT INTO test_order_where(department, salary, years, age, name) VALUES 
    ('Sales', 50000, 3, 28, 'Alice'),
    ('IT', 60000, 2, 25, 'Bob'),
    ('Sales', 45000, 5, 32, 'Carol'),
    ('IT', 65000, 4, 30, 'David'),
    ('Sales', 50000, 1, 24, 'Eve'),
    ('IT', 60000, 6, 35, 'Frank'),
    ('HR', 55000, 3, 29, 'Grace'),
    ('HR', 48000, 2, 26, 'Henry');

-- Test WHERE with ORDER BY on same column
SELECT name, salary FROM test_order_where WHERE salary > 50000 ORDER BY salary;

-- Test WHERE with ORDER BY on different column
SELECT name, department, salary FROM test_order_where WHERE salary >= 50000 ORDER BY name;

-- Test WHERE with multiple conditions and ORDER BY
SELECT name, department, years FROM test_order_where WHERE years > 2 AND department = 'IT' ORDER BY years;

-- Test WHERE with ORDER BY DESC
SELECT name, age, salary FROM test_order_where WHERE age < 30 ORDER BY salary DESC;

-- Test WHERE with ORDER BY multiple columns
SELECT name, department, salary, years FROM test_order_where WHERE salary >= 45000 ORDER BY department, salary DESC;

-- Test WHERE with comparison and ORDER BY expression
SELECT name, salary, years, (salary * years) AS total_earned 
FROM test_order_where WHERE years >= 3 ORDER BY (salary * years);

-- Test WHERE on text field with ORDER BY
SELECT name, department, age FROM test_order_where WHERE department IN ('IT', 'Sales') ORDER BY department, age;

-- Test WHERE with inequality and ORDER BY
SELECT name, salary, age FROM test_order_where WHERE salary <> 50000 ORDER BY age, salary;

DROP TABLE test_order_where;