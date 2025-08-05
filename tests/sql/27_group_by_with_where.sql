-- Test GROUP BY combined with WHERE clause for pre-aggregation filtering
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS product_orders;

CREATE TABLE product_orders (
    id SERIAL PRIMARY KEY,
    product_name VARCHAR(30),
    category VARCHAR(20),
    order_amount DECIMAL(10,2),
    quantity INTEGER,
    customer_type VARCHAR(15),
    order_date DATE,
    region VARCHAR(20)
);

INSERT INTO product_orders(product_name, category, order_amount, quantity, customer_type, order_date, region) VALUES 
    ('Laptop Pro', 'Electronics', 1500.00, 1, 'Premium', '2024-01-01', 'North'),
    ('Office Chair', 'Furniture', 200.00, 3, 'Business', '2024-01-02', 'South'),
    ('Smartphone', 'Electronics', 800.00, 2, 'Regular', '2024-01-03', 'North'),
    ('Desk Lamp', 'Furniture', 45.00, 5, 'Regular', '2024-01-04', 'East'),
    ('Gaming Mouse', 'Electronics', 75.00, 4, 'Premium', '2024-01-05', 'West'),
    ('Bookshelf', 'Furniture', 150.00, 2, 'Business', '2024-01-06', 'South'),
    ('Tablet', 'Electronics', 400.00, 1, 'Regular', '2024-01-07', 'North'),
    ('Filing Cabinet', 'Furniture', 300.00, 1, 'Business', '2024-01-08', 'East'),
    ('Headphones', 'Electronics', 120.00, 6, 'Premium', '2024-01-09', 'West'),
    ('Conference Table', 'Furniture', 800.00, 1, 'Business', '2024-01-10', 'South');

-- Test WHERE + GROUP BY with COUNT - count Electronics orders over $100
-- Should trigger MLIR compilation with FilterOp, GroupByOp, and aggregate operators
SELECT category, COUNT(*) AS order_count
FROM product_orders 
WHERE category = 'Electronics' AND order_amount > 100
GROUP BY category;

-- Test WHERE + GROUP BY with SUM - total sales by region for Premium customers
SELECT region, SUM(order_amount) AS total_sales
FROM product_orders 
WHERE customer_type = 'Premium'
GROUP BY region 
ORDER BY region;

-- Test WHERE + GROUP BY with multiple conditions and aggregates
SELECT category, 
       COUNT(*) AS order_count,
       SUM(order_amount) AS total_sales,
       AVG(quantity) AS avg_quantity
FROM product_orders 
WHERE order_amount >= 100 AND quantity <= 3
GROUP BY category 
ORDER BY category;

-- Test WHERE with date filtering + GROUP BY
SELECT customer_type, 
       COUNT(*) AS order_count,
       SUM(order_amount) AS total_sales
FROM product_orders 
WHERE order_date >= '2024-01-05'
GROUP BY customer_type 
ORDER BY customer_type;

-- Test complex WHERE conditions + GROUP BY + multiple aggregates
SELECT region, 
       category,
       COUNT(*) AS order_count,
       SUM(order_amount) AS total_sales,
       MIN(order_amount) AS min_order,
       MAX(order_amount) AS max_order
FROM product_orders 
WHERE (category = 'Electronics' OR category = 'Furniture') 
  AND order_amount BETWEEN 100 AND 1000
GROUP BY region, category 
ORDER BY region, category;

-- Test WHERE + GROUP BY + HAVING combination
SELECT customer_type, 
       COUNT(*) AS order_count,
       AVG(order_amount) AS avg_order_amount
FROM product_orders 
WHERE quantity >= 2
GROUP BY customer_type 
HAVING COUNT(*) >= 2
ORDER BY customer_type;

DROP TABLE product_orders;