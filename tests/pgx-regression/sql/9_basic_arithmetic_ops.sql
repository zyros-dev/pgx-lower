LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_arithmetic;

CREATE TABLE test_arithmetic
(
    id   SERIAL PRIMARY KEY,
    val1 INTEGER,
    val2 INTEGER
);

INSERT INTO test_arithmetic(val1, val2)
VALUES (10, 5),
       (20, 4),
       (15, 3),
       (8, 2),
       (25, 6);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_9_basic_arithmetic_ops_001 */
SELECT val1 + val2 AS addition
FROM test_arithmetic;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_9_basic_arithmetic_ops_002 */
SELECT val1 - val2 AS subtraction
FROM test_arithmetic;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_9_basic_arithmetic_ops_003 */
SELECT val1 * val2 AS multiplication
FROM test_arithmetic;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_9_basic_arithmetic_ops_004 */
SELECT val1 / val2 AS division
FROM test_arithmetic;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_9_basic_arithmetic_ops_005 */
SELECT val1 % val2 AS modulo
FROM test_arithmetic;

DROP TABLE test_arithmetic;