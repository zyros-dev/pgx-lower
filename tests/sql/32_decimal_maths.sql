LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS test_decimal_maths;

CREATE TABLE test_decimal_maths
(
    id       SERIAL PRIMARY KEY,
    price    DECIMAL(10, 2),
    quantity DECIMAL(8, 3),
    rate     DECIMAL(5, 4),
    discount DECIMAL(4, 2)
);

INSERT INTO test_decimal_maths(price, quantity, rate, discount)
VALUES (99.99, 2.500, 0.1575, 10.50),
       (149.95, 1.750, 0.2250, 15.00),
       (29.50, 5.333, 0.0825, 5.25),
       (199.00, 0.500, 0.3000, 20.00),
       (75.25, 3.125, 0.1000, 12.75);

-- Basic arithmetic operations
SELECT price + discount AS price_plus_discount
FROM test_decimal_maths;

SELECT price - discount AS price_minus_discount
FROM test_decimal_maths;

SELECT price * quantity AS total_value
FROM test_decimal_maths;

SELECT price / quantity AS price_per_unit
FROM test_decimal_maths;

-- Mixed precision operations
SELECT price * rate AS tax_amount
FROM test_decimal_maths;

SELECT (price * quantity) AS total_before_discount
FROM test_decimal_maths;

SELECT (price * quantity * (1 - discount / 100)) AS total_after_discount
FROM test_decimal_maths;

-- Division with different scales
SELECT discount / 100 AS discount_rate
FROM test_decimal_maths;

SELECT price / 2 AS half_price
FROM test_decimal_maths;

SELECT quantity / 3 AS third_quantity
FROM test_decimal_maths;

-- Complex expressions
SELECT ((price + 10) * quantity) / (1 + rate) AS complex_calc
FROM test_decimal_maths;

SELECT price * quantity * rate AS tax_on_total
FROM test_decimal_maths;

-- Edge cases
SELECT 1.0 / 3.0 AS one_third;
SELECT 10.0 / 3.0 AS ten_thirds;
SELECT 100.00 / 7.00 AS hundred_sevenths;

-- Operations with integer literals
SELECT price * 2 AS double_price
FROM test_decimal_maths;

SELECT price / 10 AS tenth_price
FROM test_decimal_maths;

SELECT quantity + 1 AS quantity_plus_one
FROM test_decimal_maths;

-- Precision preservation tests
SELECT CAST(1.23456789 AS DECIMAL(10, 8)) AS high_precision;
SELECT CAST(1.23456789 AS DECIMAL(10, 2)) AS low_precision;

-- Division by small numbers (precision test)
SELECT 1.00 / 0.01 AS hundred;
SELECT 1.00 / 0.001 AS thousand;
SELECT 10.00 / 0.1 AS hundred_alt;

-- Large number operations
SELECT CAST(999999.99 AS DECIMAL(10, 2)) * CAST(0.01 AS DECIMAL(5, 2)) AS large_times_small;
SELECT CAST(999999.99 AS DECIMAL(10, 2)) / CAST(1000.00 AS DECIMAL(10, 2)) AS large_div_thousand;

DROP TABLE test_decimal_maths;