LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS all_types_test;

-- Test subset operations with expanded type foundation
CREATE TABLE all_types_test
(
    id              SERIAL PRIMARY KEY,
    integer_col     INTEGER,           -- Core integer type
    big_int         BIGINT,            -- 64-bit integer type  
    boolean_col     BOOLEAN,           -- Boolean type
    decimal_col     INTEGER,           -- Stored as cents
    text_col        TEXT,              -- Variable-length text
    varchar_col     VARCHAR(50)        -- Variable-length string with limit
);

-- Insert deterministic test data including string columns
INSERT INTO all_types_test (integer_col, big_int, boolean_col, decimal_col, text_col, varchar_col)
SELECT 
    (1000 + i),                        -- integer_col: offset sequence
    (1000000000000 + i)::BIGINT,       -- big_int: large integers
    i % 2 = 0,                         -- boolean_col: alternating pattern
    (10000 + i * 100),                 -- decimal_col: stored as cents
    'Text data ' || i,                 -- text_col: variable text
    'Item-' || LPAD(i::TEXT, 3, '0')   -- varchar_col: formatted identifiers
FROM generate_series(1, 3) AS s(i);

-- Test basic integer selections
SELECT id, integer_col, big_int FROM all_types_test;

-- Test boolean and string values
SELECT id, boolean_col, text_col, varchar_col FROM all_types_test;

-- Test comprehensive selection - demonstrate all types working together
SELECT id, integer_col, big_int, boolean_col, decimal_col, text_col, varchar_col
FROM all_types_test;

DROP TABLE all_types_test;