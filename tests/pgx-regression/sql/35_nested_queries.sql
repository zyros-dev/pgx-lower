LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS inventory;

CREATE TABLE products
(
    product_id INTEGER,
    name       TEXT,
    category_id INTEGER,
    price      INTEGER
);

CREATE TABLE categories
(
    category_id INTEGER,
    name        TEXT,
    active      INTEGER
);

CREATE TABLE inventory
(
    product_id INTEGER,
    quantity   INTEGER,
    warehouse_id INTEGER
);

INSERT INTO products(product_id, name, category_id, price)
VALUES (1, 'Laptop', 10, 1000),
       (2, 'Mouse', 20, 25),
       (3, 'Keyboard', 20, 75),
       (4, 'Monitor', 10, 300),
       (5, 'Desk', 30, 500),
       (6, 'Chair', 30, 200);

INSERT INTO categories(category_id, name, active)
VALUES (10, 'Electronics', 1),
       (20, 'Accessories', 1),
       (30, 'Furniture', 1),
       (40, 'Discontinued', 0);

INSERT INTO inventory(product_id, quantity, warehouse_id)
VALUES (1, 50, 1),
       (2, 200, 1),
       (3, 150, 2),
       (4, 75, 1),
       (5, 30, 2);

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_35_nested_queries_001 */
SELECT name
FROM products
WHERE category_id = ANY(ARRAY(SELECT category_id FROM categories WHERE active = 1))
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_35_nested_queries_002 */
SELECT name, price
FROM products
WHERE price > (SELECT AVG(price)
               FROM products)
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_35_nested_queries_003 */
SELECT name, price
FROM products
WHERE price > ALL (SELECT price
                   FROM products
                   WHERE category_id = 20)
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_35_nested_queries_004 */
SELECT name, price
FROM products p
WHERE price > (SELECT AVG(price)
               FROM products
               WHERE category_id = p.category_id)
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_35_nested_queries_005 */
SELECT name
FROM products
WHERE category_id IN (SELECT category_id FROM categories WHERE active = 1)
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_35_nested_queries_006 */
SELECT name
FROM products
WHERE category_id NOT IN (SELECT category_id FROM categories WHERE active = 0)
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_35_nested_queries_007 */
SELECT name
FROM categories c
WHERE EXISTS (SELECT 1 FROM products WHERE category_id = c.category_id)
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_35_nested_queries_008 */
SELECT name
FROM categories c
WHERE NOT EXISTS (SELECT 1 FROM products WHERE category_id = c.category_id)
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_35_nested_queries_009 */
SELECT product_id, name,
       (SELECT name
        FROM categories c
        WHERE c.category_id = p.category_id) as category_name
FROM products p
ORDER BY product_id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_35_nested_queries_010 */
SELECT name, price,
       (SELECT MAX(quantity)
        FROM inventory i
        WHERE i.product_id = p.product_id) as max_stock
FROM products p
ORDER BY name;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_35_nested_queries_011 */
SELECT c.name, stats.total_price
FROM categories c
JOIN (SELECT category_id, SUM(price) as total_price
      FROM products
      GROUP BY category_id) AS stats
ON c.category_id = stats.category_id
ORDER BY c.name;

DROP TABLE inventory;
DROP TABLE categories;
DROP TABLE products;
