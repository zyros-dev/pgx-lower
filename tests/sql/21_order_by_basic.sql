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

DROP TABLE IF EXISTS test_order_string;
CREATE TABLE test_order_string
(
    id   SERIAL PRIMARY KEY,
    name VARCHAR(20)
);

INSERT INTO test_order_string(name)
VALUES ('Charlie'),
       ('Alice'),
       ('Bob'),
       ('David'),
       ('Eve');

SET client_min_messages TO DEBUG1;

SELECT name
FROM test_order_string
ORDER BY name;

SET client_min_messages TO NOTICE;

SELECT name
FROM test_order_string
ORDER BY name DESC;