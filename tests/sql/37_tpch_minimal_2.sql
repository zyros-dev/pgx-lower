LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS test_orders CASCADE;
DROP TABLE IF EXISTS test_items CASCADE;
DROP TABLE IF EXISTS test_suppliers CASCADE;

CREATE TABLE test_orders (
    o_id INTEGER PRIMARY KEY,
    o_date DATE NOT NULL,
    o_year INTEGER
);

CREATE TABLE test_items (
    i_id INTEGER PRIMARY KEY,
    i_order_id INTEGER NOT NULL,
    i_price DECIMAL(12,2) NOT NULL,
    i_discount DECIMAL(12,2) NOT NULL,
    i_quantity DECIMAL(12,2) NOT NULL,
    i_shipmode CHAR(10) NOT NULL
);

CREATE TABLE test_suppliers (
    s_id INTEGER PRIMARY KEY,
    s_item_id INTEGER NOT NULL,
    s_cost DECIMAL(12,2) NOT NULL
);

INSERT INTO test_orders VALUES
    (1, '1994-01-15', 1994),
    (2, '1995-06-20', 1995),
    (3, '1996-03-10', 1996);

INSERT INTO test_items VALUES
    (1, 1, 100.00, 0.05, 10.00, 'MAIL'),
    (2, 2, 200.00, 0.07, 20.00, 'SHIP'),
    (3, 3, 150.00, 0.10, 15.00, 'TRUCK');

INSERT INTO test_suppliers VALUES
    (1, 1, 50.00),
    (2, 2, 75.00),
    (3, 3, 60.00);

-- Test Case 1: Decimal cast precision issue (from Query 6)
-- This should fail with: "Failed to lower db.cast operation"
SELECT sum(i_price * i_discount) as revenue
FROM test_items
WHERE i_discount BETWEEN 0.05 - 0.01 AND 0.05 + 0.01
  AND i_quantity < 24;

-- Test Case 2: OUTER_VAR with extract(year) in subquery (from Query 7)
-- This should fail with: "OUTER_VAR varattno out of range"
SELECT o_year, sum(volume) as revenue
FROM (
    SELECT extract(year from o_date) as o_year,
           i_price * (1 - i_discount) as volume
    FROM test_orders, test_items
    WHERE o_id = i_order_id
) as subq
GROUP BY o_year
ORDER BY o_year;

-- Test Case 3: OUTER_VAR with arithmetic in nested query (from Query 9)
-- This should fail with: "OUTER_VAR varattno out of range"
SELECT o_year, sum(profit) as total_profit
FROM (
    SELECT extract(year from o_date) as o_year,
           i_price * (1 - i_discount) - s_cost * i_quantity as profit
    FROM test_orders, test_items, test_suppliers
    WHERE o_id = i_order_id
      AND s_item_id = i_id
) as profit_calc
GROUP BY o_year
ORDER BY o_year;

-- Test Case 4: InitPlan/Param in HAVING clause (from Query 11)
-- This should fail with: "Param references unknown InitPlan result"
SELECT i_id, sum(s_cost * i_quantity) as value
FROM test_suppliers, test_items
WHERE s_item_id = i_id
GROUP BY i_id
HAVING sum(s_cost * i_quantity) > (
    SELECT sum(s_cost * i_quantity) * 0.5
    FROM test_suppliers, test_items
    WHERE s_item_id = i_id
)
ORDER BY value DESC;

-- Test Case 5: ScalarArrayOpExpr with IN operator (from Query 12)
-- This should crash with: "Unsupported const array type"
SELECT i_shipmode, count(*) as cnt
FROM test_items, test_orders
WHERE o_id = i_order_id
  AND i_shipmode IN ('MAIL', 'SHIP')
GROUP BY i_shipmode
ORDER BY i_shipmode;

DROP TABLE test_orders CASCADE;
DROP TABLE test_items CASCADE;
DROP TABLE test_suppliers CASCADE;