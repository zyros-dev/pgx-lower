LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS simple_group_by;

CREATE TABLE simple_group_by
(
    id         SERIAL PRIMARY KEY,
    amount     INTEGER NOT NULL,
    quantity   INTEGER NOT NULL
);

INSERT INTO simple_group_by(amount, quantity)
VALUES (1200.00, 2),
       (25.50, 3),
       (85.75, 2),
       (29.99, 3),
       (49.99, 2),
       (300.00, 3),
       (89.99, 2),
       (15.99, 2),
       (125.00, 3),
       (450.00, 2);

SELECT quantity, COUNT(*) AS item_count
FROM simple_group_by
GROUP BY quantity;

SELECT quantity, COUNT(*) AS item_count
FROM simple_group_by
GROUP BY quantity
ORDER BY quantity;

SELECT quantity, COUNT(*) AS item_count
FROM simple_group_by
GROUP BY quantity
ORDER BY COUNT(*);

DROP TABLE IF EXISTS sales_data;

CREATE TABLE sales_data
(
    id         SERIAL PRIMARY KEY,
    department VARCHAR(20),
    product    VARCHAR(30),
    amount     INTEGER,
    quantity   INTEGER
);

INSERT INTO sales_data(department, product, amount, quantity)
VALUES ('Electronics', 'Laptop', 1200, 2),
       ('Electronics', 'Mouse', 25, 10),
       ('Electronics', 'Keyboard', 85, 5),
       ('Clothing', 'Shirt', 29, 15),
       ('Clothing', 'Pants', 49, 8),
       ('Electronics', 'Monitor', 300, 3),
       ('Clothing', 'Shoes', 89, 6),
       ('Books', 'Novel', 15, 20),
       ('Books', 'Textbook', 125, 4),
       ('Electronics', 'Tablet', 450, 2);

SELECT department, COUNT(*) AS item_count
FROM sales_data
GROUP BY department;
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
