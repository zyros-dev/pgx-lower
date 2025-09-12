LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_order_expr;

CREATE TABLE test_order_expr
(
    id          SERIAL PRIMARY KEY,
    base_salary INTEGER,
    bonus       INTEGER,
    years       INTEGER,
    name        VARCHAR(50)
);

INSERT INTO test_order_expr(base_salary, bonus, years, name)
VALUES (40000, 5000, 2, 'Alice'),
       (50000, 3000, 4, 'Bob'),
       (35000, 8000, 1, 'Carol'),
       (60000, 2000, 3, 'David'),
       (45000, 6000, 5, 'Eve');

SELECT 'lingodb doesnt support expressions in sort nodes so we dont have to either'

-- SELECT name, base_salary, bonus, (base_salary + bonus) AS total_comp
-- FROM test_order_expr
-- ORDER BY (base_salary + bonus);
--
-- SELECT name, base_salary, bonus, (base_salary + bonus) AS total_comp
-- FROM test_order_expr
-- ORDER BY (base_salary + bonus) DESC;
-- SELECT name, base_salary, bonus, years, (base_salary + bonus) AS total_comp, (base_salary * years) AS experience_value
-- FROM test_order_expr
-- ORDER BY (base_salary + bonus), (base_salary * years);
--
-- SELECT name, base_salary, bonus, years
-- FROM test_order_expr
-- ORDER BY (base_salary + bonus), years DESC;
-- SELECT name, base_salary, bonus, years, (base_salary + bonus) / years AS comp_per_year
-- FROM test_order_expr
-- ORDER BY (base_salary + bonus) / years;
-- SELECT name, base_salary, years, base_salary % 1000 AS remainder
-- FROM test_order_expr
-- ORDER BY base_salary % 1000;
-- DROP TABLE test_order_expr;
