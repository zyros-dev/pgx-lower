LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS comprehensive_data;

CREATE TABLE comprehensive_data
(
    id                SERIAL PRIMARY KEY,
    department        VARCHAR(25),
    employee_name     VARCHAR(30),
    salary            DECIMAL(10, 2),
    bonus             DECIMAL(8, 2),
    years_experience  INTEGER,
    performance_score DECIMAL(3, 1),
    hire_date         DATE,
    is_manager        BOOLEAN
);

INSERT INTO comprehensive_data(department, employee_name, salary, bonus, years_experience, performance_score, hire_date,
                               is_manager)
VALUES ('Engineering', 'Alice Johnson', 95000.00, 8000.00, 5, 4.2, '2019-03-15', false),
       ('Engineering', 'Bob Smith', 105000.00, 12000.00, 8, 4.5, '2016-07-22', true),
       ('Sales', 'Carol Davis', 75000.00, 15000.00, 3, 4.8, '2021-01-10', false),
       ('Sales', 'David Wilson', 85000.00, 18000.00, 6, 4.6, '2018-05-14', true),
       ('Marketing', 'Eve Brown', 70000.00, 5000.00, 2, 3.9, '2022-09-01', false),
       ('Engineering', 'Frank Miller', 92000.00, 7500.00, 4, 4.1, '2020-02-28', false),
       ('HR', 'Grace Lee', 65000.00, 4000.00, 7, 4.3, '2017-11-05', true),
       ('Sales', 'Henry Taylor', 78000.00, 14000.00, 4, 4.4, '2020-06-18', false),
       ('Marketing', 'Iris Chen', 72000.00, 6000.00, 3, 4.0, '2021-08-12', false),
       ('Engineering', 'Jack Anderson', 110000.00, 13000.00, 10, 4.7, '2014-04-30', true);

-- will need to write more, but you get the idea...
SELECT SUM(salary + bonus) AS total_compensation FROM comprehensive_data;
