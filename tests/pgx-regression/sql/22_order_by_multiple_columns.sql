LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_order_multi;

CREATE TABLE test_order_multi
(
    id         SERIAL PRIMARY KEY,
    department VARCHAR(20),
    salary     INTEGER,
    years      INTEGER,
    name       VARCHAR(50)
);

INSERT INTO test_order_multi(department, salary, years, name)
VALUES ('Sales', 50000, 3, 'Alice'),
       ('IT', 60000, 2, 'Bob'),
       ('Sales', 45000, 5, 'Carol'),
       ('IT', 65000, 4, 'David'),
       ('Sales', 50000, 1, 'Eve'),
       ('IT', 60000, 6, 'Frank'),
       ('HR', 55000, 3, 'Grace');

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_22_order_by_multiple_columns_001 */
SELECT department, salary, name
FROM test_order_multi
ORDER BY department, salary;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_22_order_by_multiple_columns_002 */
SELECT department, salary, name
FROM test_order_multi
ORDER BY department ASC, salary ASC;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_22_order_by_multiple_columns_003 */
SELECT department, salary, name
FROM test_order_multi
ORDER BY department ASC, salary DESC;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_22_order_by_multiple_columns_004 */
SELECT department, salary, name
FROM test_order_multi
ORDER BY department DESC, salary DESC;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_22_order_by_multiple_columns_005 */
SELECT department, salary, years, name
FROM test_order_multi
ORDER BY department, salary, years;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_22_order_by_multiple_columns_006 */
SELECT department, salary, years, name
FROM test_order_multi
ORDER BY department ASC, salary DESC, years ASC;
DROP TABLE test_order_multi;
