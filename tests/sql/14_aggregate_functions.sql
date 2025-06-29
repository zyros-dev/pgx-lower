-- Test aggregate functions: PgSumOp, PgCountOp, PgAvgOp, PgMinOp, PgMaxOp
LOAD 'pgx_lower';

DROP TABLE IF EXISTS test_aggregates;

CREATE TABLE test_aggregates (
    id SERIAL PRIMARY KEY,
    category VARCHAR(20),
    amount DECIMAL(10,2),
    quantity INTEGER,
    price REAL
);

INSERT INTO test_aggregates(category, amount, quantity, price) VALUES 
    ('A', 100.50, 5, 20.10),
    ('B', 200.75, 3, 66.92),
    ('A', 150.25, 7, 21.46),
    ('C', 300.00, 2, 150.00),
    ('B', 175.80, 4, 43.95),
    ('A', 125.60, 6, 20.93),
    ('C', 250.40, 1, 250.40);

-- Test aggregate functions in SELECT clauses (with GROUP BY)
-- These should trigger MLIR compilation with aggregate operators
SELECT category, SUM(amount) AS total_amount FROM test_aggregates GROUP BY category;
SELECT category, COUNT(*) AS row_count FROM test_aggregates GROUP BY category;
SELECT category, COUNT(quantity) AS quantity_count FROM test_aggregates GROUP BY category;
SELECT category, AVG(amount) AS avg_amount FROM test_aggregates GROUP BY category;
SELECT category, MIN(amount) AS min_amount FROM test_aggregates GROUP BY category;
SELECT category, MAX(amount) AS max_amount FROM test_aggregates GROUP BY category;
SELECT category, MIN(price) AS min_price FROM test_aggregates GROUP BY category;
SELECT category, MAX(price) AS max_price FROM test_aggregates GROUP BY category;

-- Test aggregate functions without GROUP BY (whole table aggregates)
-- These should trigger MLIR compilation with aggregate operators
SELECT SUM(amount) AS total_amount_all FROM test_aggregates;
SELECT COUNT(*) AS total_rows FROM test_aggregates;
SELECT COUNT(DISTINCT category) AS unique_categories FROM test_aggregates;
SELECT AVG(amount) AS avg_amount_all FROM test_aggregates;
SELECT MIN(amount) AS min_amount_all FROM test_aggregates;
SELECT MAX(amount) AS max_amount_all FROM test_aggregates;
SELECT SUM(quantity) AS total_quantity FROM test_aggregates;
SELECT AVG(price) AS avg_price FROM test_aggregates;

-- Test aggregate functions in HAVING clauses
-- These should trigger MLIR compilation with aggregate operators in predicates
SELECT category, SUM(amount) AS total FROM test_aggregates GROUP BY category HAVING SUM(amount) > 200;
SELECT category, COUNT(*) AS cnt FROM test_aggregates GROUP BY category HAVING COUNT(*) > 1;
SELECT category, AVG(amount) AS avg FROM test_aggregates GROUP BY category HAVING AVG(amount) > 150;
SELECT category, MIN(price) AS min_p FROM test_aggregates GROUP BY category HAVING MIN(price) < 25;
SELECT category, MAX(quantity) AS max_q FROM test_aggregates GROUP BY category HAVING MAX(quantity) > 5;

DROP TABLE test_aggregates;