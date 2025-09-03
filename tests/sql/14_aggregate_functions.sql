LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_aggregates;

CREATE TABLE test_aggregates
(
    id       SERIAL PRIMARY KEY,
    category INTEGER,
    amount   INTEGER,
    quantity INTEGER,
    price    INTEGER
);

INSERT INTO test_aggregates(category, amount, quantity, price)
VALUES (1, 10050, 5, 2010),
       (2, 20075, 3, 6692),
       (1, 15025, 7, 2146),
       (3, 30000, 2, 15000),
       (2, 17580, 4, 4395),
       (1, 12560, 6, 2093),
       (3, 25040, 1, 25040);

SELECT SUM(amount) AS total_amount_all
FROM test_aggregates;
SELECT COUNT(*) AS total_rows
FROM test_aggregates;
SELECT COUNT(DISTINCT category) AS unique_categories
FROM test_aggregates;
SELECT AVG(amount) AS avg_amount_all
FROM test_aggregates;
SELECT MIN(amount) AS min_amount_all
FROM test_aggregates;
SELECT MAX(amount) AS max_amount_all
FROM test_aggregates;
SELECT SUM(quantity) AS total_quantity
FROM test_aggregates;
SELECT AVG(price) AS avg_price
FROM test_aggregates;

DROP TABLE test_aggregates;