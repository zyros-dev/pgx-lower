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

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_001 */
-- Basic arithmetic operations
SELECT price + discount AS price_plus_discount
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_002 */
SELECT price - discount AS price_minus_discount
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_003 */
SELECT price * quantity AS total_value
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_004 */
SELECT price / quantity AS price_per_unit
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_005 */
-- Mixed precision operations
SELECT price * rate AS tax_amount
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_006 */
SELECT (price * quantity) AS total_before_discount
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_007 */
SELECT (price * quantity * (1 - discount / 100)) AS total_after_discount
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_008 */
-- Division with different scales
SELECT discount / 100 AS discount_rate
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_009 */
SELECT price / 2 AS half_price
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_010 */
SELECT quantity / 3 AS third_quantity
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_011 */
-- Complex expressions
SELECT ((price + 10) * quantity) / (1 + rate) AS complex_calc
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_012 */
SELECT price * quantity * rate AS tax_on_total
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_013 */
-- Edge cases
SELECT 1.0 / 3.0 AS one_third;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_014 */
SELECT 10.0 / 3.0 AS ten_thirds;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_015 */
SELECT 100.00 / 7.00 AS hundred_sevenths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_016 */
-- Operations with integer literals
SELECT price * 2 AS double_price
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_017 */
SELECT price / 10 AS tenth_price
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_32_decimal_maths_018 */
SELECT quantity + 1 AS quantity_plus_one
FROM test_decimal_maths;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_019 */
-- Precision preservation tests
SELECT CAST(1.23456789 AS DECIMAL(10, 8)) AS high_precision;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_020 */
SELECT CAST(1.23456789 AS DECIMAL(10, 2)) AS low_precision;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_021 */
-- Division by small numbers (precision test)
SELECT 1.00 / 0.01 AS hundred;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_022 */
SELECT 1.00 / 0.001 AS thousand;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_023 */
SELECT 10.00 / 0.1 AS hundred_alt;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_024 */
-- Large number operations
SELECT CAST(999999.99 AS DECIMAL(10, 2)) * CAST(0.01 AS DECIMAL(5, 2)) AS large_times_small;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_32_decimal_maths_025 */
SELECT CAST(999999.99 AS DECIMAL(10, 2)) / CAST(1000.00 AS DECIMAL(10, 2)) AS large_div_thousand;

DROP TABLE test_decimal_maths;