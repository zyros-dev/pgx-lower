LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS all_types_test;

CREATE TABLE all_types_test
(
    id          SERIAL PRIMARY KEY,
    integer_col INTEGER,
    big_int     BIGINT,
    boolean_col BOOLEAN,
    decimal_col INTEGER,
    text_col    TEXT,
    varchar_col VARCHAR(50)
);

INSERT INTO all_types_test (integer_col, big_int, boolean_col, decimal_col, text_col, varchar_col)
SELECT (1000 + i),
       (1000000000000 + i)::BIGINT, i % 2 = 0,
    (10000 + i * 100),
    'Text data ' || i,
    'Item-' || LPAD(i::TEXT, 3, '0')
FROM generate_series(1, 3) AS s(i);

SELECT id, integer_col, big_int
FROM all_types_test;

SELECT id, boolean_col, text_col, varchar_col
FROM all_types_test;

SELECT id, integer_col, big_int, boolean_col, decimal_col, text_col, varchar_col
FROM all_types_test;

DROP TABLE all_types_test;