LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_order_basic;

CREATE TABLE test_order_basic
(
    id    SERIAL PRIMARY KEY,
    value INTEGER,
    name  INTEGER
);

INSERT INTO test_order_basic(value, name)
VALUES (30, 3),
       (10, 1),
       (20, 2),
       (40, 4),
       (15, 5);

SELECT value, name
FROM test_order_basic
ORDER BY value;
SELECT value, name
FROM test_order_basic
ORDER BY value ASC;
SELECT value, name
FROM test_order_basic
ORDER BY value DESC;
SELECT value, name
FROM test_order_basic
ORDER BY name;
DROP TABLE test_order_basic;
