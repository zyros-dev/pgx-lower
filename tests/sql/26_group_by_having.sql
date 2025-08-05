-- Test GROUP BY with HAVING clause for post-aggregation filtering
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS employee_sales;

CREATE TABLE employee_sales (
    id SERIAL PRIMARY KEY,
    employee_name VARCHAR(30),
    department VARCHAR(20),
    sale_amount DECIMAL(10,2),
    commission_rate DECIMAL(4,2),
    sale_date DATE
);

INSERT INTO employee_sales(employee_name, department, sale_amount, commission_rate, sale_date) VALUES 
    ('Alice', 'Sales', 5000.00, 0.10, '2024-01-01'),
    ('Bob', 'Sales', 3500.00, 0.08, '2024-01-02'),
    ('Carol', 'Marketing', 2000.00, 0.12, '2024-01-03'),
    ('Alice', 'Sales', 4500.00, 0.10, '2024-01-04'),
    ('David', 'IT', 1500.00, 0.15, '2024-01-05'),
    ('Bob', 'Sales', 6000.00, 0.08, '2024-01-06'),
    ('Carol', 'Marketing', 3000.00, 0.12, '2024-01-07'),
    ('Eve', 'IT', 2500.00, 0.15, '2024-01-08'),
    ('Alice', 'Sales', 5500.00, 0.10, '2024-01-09'),
    ('David', 'IT', 1800.00, 0.15, '2024-01-10');

-- Test HAVING with COUNT - departments with more than 2 sales
-- Should trigger MLIR compilation with GroupByOp, aggregate operators, and HAVING filter
SELECT department, COUNT(*) AS sale_count
FROM employee_sales 
GROUP BY department 
HAVING COUNT(*) > 2
ORDER BY department;

-- Test HAVING with SUM - employees with total sales over 8000
SELECT employee_name, SUM(sale_amount) AS total_sales
FROM employee_sales 
GROUP BY employee_name 
HAVING SUM(sale_amount) > 8000
ORDER BY employee_name;

-- Test HAVING with AVG - departments with average sale over 3000
SELECT department, AVG(sale_amount) AS avg_sale
FROM employee_sales 
GROUP BY department 
HAVING AVG(sale_amount) > 3000
ORDER BY department;

-- Test HAVING with multiple conditions
SELECT department, 
       COUNT(*) AS sale_count,
       SUM(sale_amount) AS total_sales,
       AVG(sale_amount) AS avg_sale
FROM employee_sales 
GROUP BY department 
HAVING COUNT(*) >= 2 AND SUM(sale_amount) > 5000
ORDER BY department;

-- Test HAVING with MIN/MAX conditions
SELECT employee_name, 
       COUNT(*) AS sale_count,
       MIN(sale_amount) AS min_sale,
       MAX(sale_amount) AS max_sale
FROM employee_sales 
GROUP BY employee_name 
HAVING MIN(sale_amount) > 2000 AND MAX(sale_amount) < 6000
ORDER BY employee_name;

DROP TABLE employee_sales;