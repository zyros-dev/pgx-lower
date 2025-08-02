-- Test WHERE clause with pattern matching: LIKE, ILIKE, IN, ANY operators
LOAD 'pgx_lower';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → SubOp → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_where_patterns;

CREATE TABLE test_where_patterns (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    email VARCHAR(100),
    department VARCHAR(20),
    salary INTEGER,
    status VARCHAR(10)
);

INSERT INTO test_where_patterns(name, email, department, salary, status) VALUES 
    ('Alice Smith', 'alice.smith@company.com', 'Engineering', 75000, 'active'),
    ('Bob Johnson', 'bob.j@company.com', 'Marketing', 65000, 'inactive'),
    ('Carol Davis', 'carol.davis@company.com', 'Engineering', 80000, 'active'),
    ('David Wilson', 'david@company.com', 'Sales', 70000, 'active'),
    ('Eve Brown', 'eve.brown@company.com', 'Marketing', 68000, 'pending'),
    ('Frank Miller', 'frank.m@company.com', 'Sales', 72000, 'active'),
    ('Grace Taylor', 'grace@company.com', 'Engineering', 85000, 'inactive');

-- Test WHERE with LIKE pattern matching
-- These should trigger MLIR compilation with string pattern operations
SELECT id, name FROM test_where_patterns WHERE name LIKE 'A%';
SELECT id, name FROM test_where_patterns WHERE name LIKE '%Smith';
SELECT id, name FROM test_where_patterns WHERE name LIKE '%o%';
SELECT id, email FROM test_where_patterns WHERE email LIKE '%.%@%';
SELECT id, email FROM test_where_patterns WHERE email LIKE '%company.com';

-- Test WHERE with LIKE and wildcards
SELECT id, name FROM test_where_patterns WHERE name LIKE '___ %';  -- Three letter first name
SELECT id, name FROM test_where_patterns WHERE name LIKE '% _____';  -- Five letter last name
SELECT department FROM test_where_patterns WHERE department LIKE '%ing';

-- Test WHERE with NOT LIKE
SELECT id, name FROM test_where_patterns WHERE name NOT LIKE 'A%';
SELECT id, email FROM test_where_patterns WHERE email NOT LIKE '%@company.com';

-- Test WHERE with IN operator
SELECT id, name, department FROM test_where_patterns WHERE department IN ('Engineering', 'Sales');
SELECT id, name, status FROM test_where_patterns WHERE status IN ('active', 'pending');
SELECT id, name FROM test_where_patterns WHERE salary IN (75000, 80000, 85000);

-- Test WHERE with NOT IN operator
SELECT id, name, department FROM test_where_patterns WHERE department NOT IN ('Marketing');
SELECT id, name, status FROM test_where_patterns WHERE status NOT IN ('inactive');

-- Test WHERE with IN and subqueries (if supported)
SELECT id, name FROM test_where_patterns WHERE department IN (SELECT DISTINCT department FROM test_where_patterns WHERE salary > 70000);

-- Test WHERE with pattern matching and logical combinations
SELECT id, name, department FROM test_where_patterns WHERE name LIKE 'C%' AND department = 'Engineering';
SELECT id, name, salary FROM test_where_patterns WHERE (name LIKE '%e%' OR name LIKE '%a%') AND salary > 70000;
SELECT id, name FROM test_where_patterns WHERE department IN ('Engineering', 'Sales') AND status = 'active';

-- Test WHERE with complex pattern combinations
SELECT id, name, email FROM test_where_patterns WHERE name LIKE '%a%' AND email LIKE '%@company.com' AND department NOT IN ('Marketing');
SELECT id, name, department FROM test_where_patterns WHERE (name NOT LIKE 'A%' AND name NOT LIKE 'B%') OR department = 'Engineering';

DROP TABLE test_where_patterns;