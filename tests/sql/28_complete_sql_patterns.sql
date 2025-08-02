-- Test complete SQL patterns with GROUP BY, ORDER BY, LIMIT, and complex expressions
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → SubOp → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS comprehensive_data;

CREATE TABLE comprehensive_data (
    id SERIAL PRIMARY KEY,
    department VARCHAR(25),
    employee_name VARCHAR(30),
    salary DECIMAL(10,2),
    bonus DECIMAL(8,2),
    years_experience INTEGER,
    performance_score DECIMAL(3,1),
    hire_date DATE,
    is_manager BOOLEAN
);

INSERT INTO comprehensive_data(department, employee_name, salary, bonus, years_experience, performance_score, hire_date, is_manager) VALUES 
    ('Engineering', 'Alice Johnson', 95000.00, 8000.00, 5, 4.2, '2019-03-15', false),
    ('Engineering', 'Bob Smith', 105000.00, 12000.00, 8, 4.5, '2016-07-22', true),
    ('Sales', 'Carol Davis', 75000.00, 15000.00, 3, 4.8, '2021-01-10', false),
    ('Sales', 'David Wilson', 85000.00, 18000.00, 6, 4.6, '2018-05-14', true),
    ('Marketing', 'Eve Brown', 70000.00, 5000.00, 2, 3.9, '2022-09-01', false),
    ('Engineering', 'Frank Miller', 92000.00, 7500.00, 4, 4.1, '2020-02-28', false),
    ('HR', 'Grace Lee', 65000.00, 4000.00, 7, 4.3, '2017-11-05', true),
    ('Sales', 'Henry Taylor', 78000.00, 14000.00, 4, 4.4, '2020-06-18', false),
    ('Marketing', 'Iris Chen', 72000.00, 6000.00, 3, 4.0, '2021-08-12', false),
    ('Engineering', 'Jack Anderson', 110000.00, 13000.00, 10, 4.7, '2014-04-30', true);

-- Test GROUP BY with calculated expressions and ORDER BY
-- Should trigger full MLIR pipeline with complex expression compilation
SELECT department, 
       COUNT(*) AS employee_count,
       ROUND(AVG(salary + bonus), 2) AS avg_total_compensation,
       SUM(salary + bonus) AS total_compensation
FROM comprehensive_data 
GROUP BY department 
ORDER BY avg_total_compensation DESC;

-- Test GROUP BY with conditional aggregation and LIMIT
SELECT department,
       COUNT(*) AS total_employees,
       COUNT(CASE WHEN is_manager THEN 1 END) AS manager_count,
       COUNT(CASE WHEN performance_score > 4.0 THEN 1 END) AS high_performers
FROM comprehensive_data 
GROUP BY department 
ORDER BY total_employees DESC 
LIMIT 3;

-- Test complex WHERE + GROUP BY + HAVING + ORDER BY
SELECT department,
       AVG(years_experience) AS avg_experience,
       AVG(performance_score) AS avg_performance,
       COUNT(*) AS employee_count
FROM comprehensive_data 
WHERE salary > 70000 AND years_experience >= 3
GROUP BY department 
HAVING COUNT(*) >= 2 AND AVG(performance_score) > 4.0
ORDER BY avg_performance DESC, avg_experience DESC;

-- Test GROUP BY with multiple grouping columns
SELECT department, 
       is_manager,
       COUNT(*) AS count,
       AVG(salary) AS avg_salary,
       MIN(hire_date) AS earliest_hire,
       MAX(hire_date) AS latest_hire
FROM comprehensive_data 
GROUP BY department, is_manager 
ORDER BY department, is_manager DESC;

-- Test advanced aggregation with mathematical expressions
SELECT department,
       COUNT(*) AS team_size,
       SUM(salary) AS total_salary_cost,
       AVG(salary + (bonus * 1.2)) AS avg_adjusted_comp,
       ROUND(SUM(salary) / COUNT(*), 2) AS salary_per_employee,
       MAX(years_experience) - MIN(years_experience) AS experience_range
FROM comprehensive_data 
WHERE performance_score >= 4.0
GROUP BY department 
HAVING COUNT(*) >= 2
ORDER BY avg_adjusted_comp DESC 
LIMIT 5;

-- Test GROUP BY with string operations and aggregation
SELECT UPPER(SUBSTRING(department, 1, 3)) AS dept_code,
       COUNT(*) AS employee_count,
       STRING_AGG(employee_name, ', ' ORDER BY salary DESC) AS employees_by_salary
FROM comprehensive_data 
WHERE LENGTH(department) > 5
GROUP BY UPPER(SUBSTRING(department, 1, 3))
ORDER BY employee_count DESC;

-- Test complex date-based grouping with aggregation
SELECT EXTRACT(YEAR FROM hire_date) AS hire_year,
       COUNT(*) AS hires_count,
       AVG(salary) AS avg_starting_salary,
       SUM(CASE WHEN is_manager THEN 1 ELSE 0 END) AS managers_hired
FROM comprehensive_data 
WHERE hire_date >= '2018-01-01'
GROUP BY EXTRACT(YEAR FROM hire_date)
HAVING COUNT(*) >= 2
ORDER BY hire_year;

DROP TABLE comprehensive_data;