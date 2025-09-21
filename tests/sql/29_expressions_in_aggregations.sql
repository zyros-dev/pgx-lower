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

SELECT SUM(salary + bonus) AS total_compensation FROM comprehensive_data;

SELECT
    AVG(salary / 1000) AS avg_salary_thousands,
    SUM(bonus * 2) AS double_bonus_total,
    MAX(salary - 50000) AS max_above_baseline
FROM comprehensive_data;

SELECT
    COUNT(salary + bonus) AS comp_count,
    MIN(years_experience * 10000) AS min_exp_value,
    MAX(years_experience + 100) AS max_exp_plus
FROM comprehensive_data;

SELECT
    department,
    AVG(salary + bonus) AS avg_total_comp,
    SUM(salary / 4) AS quarterly_salary_sum
FROM comprehensive_data
GROUP BY department
ORDER BY department;

SELECT
    SUM((salary + bonus) / 5) AS weighted_comp,
    AVG(years_experience * 1000) AS experience_value
FROM comprehensive_data;

SELECT
    AVG(salary / years_experience) AS salary_per_year_exp,
    SUM(bonus / 12) AS monthly_bonus_total,
    MAX(salary / 7) AS max_weekly_estimate
FROM comprehensive_data
WHERE years_experience > 0;

SELECT
    department,
    SUM(salary + (bonus * 2)) AS total_with_double_bonus,
    AVG(years_experience * 25) AS avg_exp_quarter
FROM comprehensive_data
WHERE salary > 70000
GROUP BY department
ORDER BY department;

SELECT
    COUNT(*) AS total_employees,
    SUM(salary - 60000) AS total_above_minimum,
    AVG((salary + bonus) / 12) AS avg_monthly_comp,
    MAX(years_experience * 100) AS max_exp_hundred,
    MIN(bonus / 100) AS min_bonus_hundred
FROM comprehensive_data;

SELECT
    department,
    AVG(salary + bonus) AS avg_compensation,
    SUM(years_experience * 1000) AS total_exp_points
FROM comprehensive_data
GROUP BY department
HAVING AVG(salary + bonus) > 80000
ORDER BY avg_compensation DESC;

SELECT
    SUM((salary + bonus) - (salary / 5)) AS net_after_tax_estimate,
    AVG(years_experience * 10000) AS exp_adjusted_value
FROM comprehensive_data
WHERE years_experience > 3;

SELECT
    SUM(salary + COALESCE(bonus, 0)) AS total_with_null_handling,
    AVG(COALESCE(bonus, 0) / COALESCE(years_experience, 1)) AS avg_bonus_per_year
FROM comprehensive_data;

SELECT
    department,
    COUNT(employee_name || ' - ' || department) AS name_dept_combinations
FROM comprehensive_data
GROUP BY department
ORDER BY department;

SELECT
    department,
    SUM(salary + bonus - 1000) AS adjusted_compensation,
    AVG(salary / 10000) AS salary_factor,
    MAX(bonus + 1000) AS max_bonus_plus_thousand
FROM comprehensive_data
GROUP BY department
ORDER BY department;

SELECT
    SUM(years_experience % 10) AS exp_remainders,
    AVG((bonus + 1000) * 2) AS adjusted_bonus_avg,
    MIN(years_experience + 10) AS min_exp_plus_ten
FROM comprehensive_data;

SELECT
    AVG(salary - 85000) AS avg_distance_from_median,
    SUM(bonus - 10000) AS total_bonus_deviation
FROM comprehensive_data;

SELECT
    ABS(AVG(salary - 85000)) AS avg_distance_from_median,
    ABS(SUM(bonus - 10000)) AS total_bonus_deviation
FROM comprehensive_data;

SELECT
    AVG(ABS(salary - 85000)) AS avg_distance_from_median,
    SUM(ABS(bonus - 10000)) AS total_bonus_deviation
FROM comprehensive_data;

DROP TABLE comprehensive_data;