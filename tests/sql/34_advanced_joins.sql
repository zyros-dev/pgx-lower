-- This test case covers LEFT, RIGHT, SEMI, and ANTI joins

LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS projects;

CREATE TABLE employees
(
    emp_id   INTEGER,
    emp_name TEXT,
    dept_id  INTEGER,
    salary   INTEGER
);

CREATE TABLE departments
(
    dept_id   INTEGER,
    dept_name TEXT,
    location  TEXT
);

CREATE TABLE projects
(
    project_id   INTEGER,
    project_name TEXT,
    dept_id      INTEGER,
    budget       INTEGER
);

INSERT INTO employees(emp_id, emp_name, dept_id, salary)
VALUES (1, 'Alice', 10, 75000),
       (2, 'Bob', 20, 65000),
       (3, 'Carol', 10, 80000),
       (4, 'David', 30, 70000),
       (5, 'Eve', NULL, 60000),
       (6, 'Frank', 40, 72000);

INSERT INTO departments(dept_id, dept_name, location)
VALUES (10, 'Engineering', 'Building A'),
       (20, 'Sales', 'Building B'),
       (30, 'HR', 'Building C'),
       (50, 'Marketing', 'Building D');

INSERT INTO projects(project_id, project_name, dept_id, budget)
VALUES (100, 'Project Alpha', 10, 500000),
       (200, 'Project Beta', 20, 300000),
       (300, 'Project Gamma', 10, 450000),
       (400, 'Project Delta', 60, 200000);

SELECT e.emp_name, e.salary, d.dept_name, d.location
FROM employees e
         LEFT JOIN departments d ON e.dept_id = d.dept_id
ORDER BY e.emp_id;

SELECT e.emp_name, e.salary, d.dept_name, d.location
FROM employees e
         RIGHT JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_id, e.emp_id;

SELECT e.emp_name, e.salary, e.dept_id
FROM employees e
WHERE EXISTS (SELECT 1
              FROM departments d
              WHERE d.dept_id = e.dept_id)
ORDER BY e.emp_id;

SELECT e.emp_name, e.salary, e.dept_id
FROM employees e
WHERE NOT EXISTS (SELECT 1
                  FROM departments d
                  WHERE d.dept_id = e.dept_id)
ORDER BY e.emp_id;



SELECT d.dept_name, d.location
FROM departments d
WHERE d.dept_id IN (SELECT p.dept_id
                    FROM projects p)
ORDER BY d.dept_id;

SELECT e.emp_name, d.dept_name
FROM employees e
         LEFT JOIN departments d ON e.dept_id = d.dept_id
WHERE d.location = 'Building A'
   OR d.location IS NULL
ORDER BY e.emp_id;

SELECT e.emp_name,
       d.dept_name,
       p.project_name,
       p.budget
FROM employees e
         LEFT JOIN departments d ON e.dept_id = d.dept_id
         LEFT JOIN projects p ON d.dept_id = p.dept_id
ORDER BY e.emp_id, p.project_id;

SELECT e1.emp_name as employee,
       e2.emp_name as colleague,
       e1.dept_id
FROM employees e1
         LEFT JOIN employees e2
                   ON e1.dept_id = e2.dept_id
                       AND e1.emp_id < e2.emp_id
ORDER BY e1.emp_id, e2.emp_id;

DROP TABLE employees;
DROP TABLE departments;
DROP TABLE projects;