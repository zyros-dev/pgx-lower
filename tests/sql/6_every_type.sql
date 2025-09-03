LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS test;

-- Test our core type foundation: INTEGER, BIGINT, BOOLEAN, SERIAL, TEXT
CREATE TABLE test
(
    id              SERIAL PRIMARY KEY,
    integer_value   INTEGER,           -- Core integer type
    big_value       BIGINT,            -- 64-bit integer type  
    boolean_flag    BOOLEAN,           -- Boolean type
    text_field      TEXT,              -- Variable-length text
    category_name   VARCHAR(20),       -- Variable-length string with limit
    amount_cents    INTEGER            -- Monetary values as cents
);

-- Insert test data including string columns
INSERT INTO test (integer_value, big_value, boolean_flag, text_field, category_name, amount_cents)
SELECT 
    (1000 + i),                            -- integer_value: standard integers  
    (1000000000000 + i)::BIGINT,           -- big_value: big integers
    i % 2 = 0,                             -- boolean_flag: alternating true/false
    'Description ' || i,                   -- text_field: variable text
    CASE i % 3 WHEN 0 THEN 'Electronics' WHEN 1 THEN 'Books' ELSE 'Toys' END,  -- category_name: string categories
    (10000 + i * 100)                      -- amount_cents: monetary values as cents
FROM generate_series(1, 5) AS s(i);

-- Test basic integer selections
SELECT id, integer_value, big_value FROM test;
-- Enable debug logging for string query - focus on AST translation

SELECT category_name, text_field FROM test;

-- Test boolean and string selections
SELECT id, boolean_flag, text_field FROM test;


-- Final comprehensive selection - test all types together
SELECT id, integer_value, big_value, boolean_flag, text_field, category_name, amount_cents
FROM test;

DROP TABLE test;
