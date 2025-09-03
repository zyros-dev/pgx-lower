LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test;

CREATE TABLE test
(
    id            SERIAL PRIMARY KEY,
    integer_value INTEGER,
    big_value     BIGINT,
    boolean_flag  BOOLEAN,
    text_field    TEXT,
    category_name VARCHAR(20),
    amount_cents  INTEGER
);

INSERT INTO test (integer_value, big_value, boolean_flag, text_field, category_name, amount_cents)
SELECT (1000 + i),
       (1000000000000 + i)::BIGINT, i % 2 = 0,
    'Description ' || i,
    CASE i % 3 WHEN 0 THEN 'Electronics' WHEN 1 THEN 'Books' ELSE 'Toys'
END
,
    (10000 + i * 100)
FROM generate_series(1, 5) AS s(i);

SELECT id, integer_value, big_value
FROM test;

SELECT category_name, text_field
FROM test;

SELECT id, boolean_flag, text_field
FROM test;

SELECT id, integer_value, big_value, boolean_flag, text_field, category_name, amount_cents
FROM test;

DROP TABLE test;
