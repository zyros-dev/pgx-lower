LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS sales_data;

CREATE TABLE sales_data
(
    id         SERIAL PRIMARY KEY,
    department VARCHAR(20),
    product    VARCHAR(30),
    amount     DECIMAL(10, 2),
    quantity   INTEGER,
    sales_date DATE
);

INSERT INTO sales_data(department, product, amount, quantity, sales_date)
VALUES ('Electronics', 'Laptop', 1200.00, 2, '2024-01-15'),
       ('Electronics', 'Mouse', 25.50, 10, '2024-01-16'),
       ('Electronics', 'Keyboard', 85.75, 5, '2024-01-17'),
       ('Clothing', 'Shirt', 29.99, 15, '2024-01-18'),
       ('Clothing', 'Pants', 49.99, 8, '2024-01-19'),
       ('Electronics', 'Monitor', 300.00, 3, '2024-01-20'),
       ('Clothing', 'Shoes', 89.99, 6, '2024-01-21'),
       ('Books', 'Novel', 15.99, 20, '2024-01-22'),
       ('Books', 'Textbook', 125.00, 4, '2024-01-23'),
       ('Electronics', 'Tablet', 450.00, 2, '2024-01-24');

SELECT department, COUNT(*) AS item_count
FROM sales_data
GROUP BY department
ORDER BY department;
SELECT department, SUM(amount) AS total_sales
FROM sales_data
GROUP BY department
ORDER BY department;
SELECT department, AVG(amount) AS avg_amount
FROM sales_data
GROUP BY department
ORDER BY department;
SELECT department, MIN(amount) AS min_amount, MAX(amount) AS max_amount
FROM sales_data
GROUP BY department
ORDER BY department;
SELECT department,
       COUNT(*)      AS item_count,
       SUM(amount)   AS total_sales,
       AVG(quantity) AS avg_quantity
FROM sales_data
GROUP BY department
ORDER BY department;
DROP TABLE sales_data;
