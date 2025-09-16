LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS type_test_table;

CREATE TABLE type_test_table (
    bool_col BOOLEAN,           -- BOOLOID 16
    int2_col SMALLINT,          -- INT2OID 21
    int4_col INTEGER,           -- INT4OID 23
    int8_col BIGINT,            -- INT8OID 20
    float4_col REAL,            -- FLOAT4OID 700
    float8_col DOUBLE PRECISION, -- FLOAT8OID 701
    string_col VARCHAR(100),    -- VARCHAROID 1043
    char_col CHAR(10),          -- BPCHAROID 1042
    text_col TEXT,              -- TEXTOID 25
    decimal_col DECIMAL(10, 2), -- NUMERICOID 1700
    numeric_col NUMERIC(15, 5), -- NUMERICOID 1700
    date_col DATE,              -- DATEOID 1082
    timestamp_col TIMESTAMP,    -- TIMESTAMPOID 1114
    interval_col INTERVAL       -- INTERVALOID 1186
);

INSERT INTO type_test_table VALUES
    (true, 100, 1000, 100000, 3.14, 3.14159265359, 'hello', 'fixed     ', 'longer text here', 12345.67, 98765.43210, '2024-01-15', '2024-01-15 10:30:00', INTERVAL '5 days'),
    (false, -200, -2000, -200000, -2.71, -2.71828182846, 'world', 'test      ', 'another text', -9999.99, -12345.67890, '2024-06-30', '2024-06-30 14:45:30.123', INTERVAL '3 months'),
    (true, 32767, 2147483647, 9223372036854775807, 1.23e10, 1.23e100, 'special!@#', 'chars$%^  ', 'unicode αβγ', 99999.99, 99999.99999, '2024-12-31', '2024-12-31 23:59:59.999999', INTERVAL '1 year 2 months'),
    (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
    (false, 0, 0, 0, 0.0, 0.0, '', '          ', '', 0.00, 0.00000, '2024-02-29', '2024-02-29 00:00:00', INTERVAL '0 days');

-- PROBLEM: Column 4 (is_null) outputs 0/1 instead of t/f for boolean results
SELECT bool_col,
       bool_col = true AS is_true,
       bool_col = false AS is_false,
       bool_col IS NULL AS is_null
FROM type_test_table;

-- PROBLEM: Column 4 (int_sum) has integer overflow - row 3: 32767 + 2147483647 = -2147450882 (wrong)
-- EXPECTED: 2147516414
SELECT int2_col > 0 AS int2_positive,
       int4_col < 1000 AS int4_small,
       int8_col >= 100000 AS int8_large,
       int2_col + int4_col AS int_sum
FROM type_test_table;

-- PROBLEM: All aggregations completely wrong
-- SUM(int2_col) returns -100001, should be 32667
-- AVG(int2_col) returns -25001, should be ~8166.75
-- Appears to be memory corruption or wrong offsets
SELECT SUM(int2_col) AS sum_int2,
       SUM(int4_col) AS sum_int4,
       SUM(int8_col) AS sum_int8,
       AVG(int2_col) AS avg_int2,
       AVG(int4_col) AS avg_int4,
       AVG(int8_col) AS avg_int8,
       MIN(int2_col) AS min_int2,
       MAX(int4_col) AS max_int4
FROM type_test_table;

-- PROBLEM: Float arithmetic mostly works but has precision issues
SELECT float4_col > 0 AS float4_positive,
       float8_col < 10 AS float8_small,
       float4_col * 2.0 AS float4_doubled,
       float8_col / 3.0 AS float8_divided
FROM type_test_table;

-- PROBLEM: Float aggregations return raw IEEE 754 bit patterns as integers
-- SUM(float4_col) returns 1076170540 (raw bits), should be ~12300000043
-- All float aggregations show integer representations of float memory
SELECT SUM(float4_col) AS sum_float4,
       SUM(float8_col) AS sum_float8,
       AVG(float4_col) AS avg_float4,
       AVG(float8_col) AS avg_float8,
       MIN(float4_col) AS min_float4,
       MAX(float8_col) AS max_float8
FROM type_test_table;

-- PROBLEM: String operations mostly work but NULL handling shows empty strings
SELECT string_col,
       string_col = 'hello' AS is_hello,
       string_col > 'a' AS after_a,
       string_col < 'z' AS before_z,
       string_col LIKE 'h%' AS starts_with_h,
       string_col || '!' AS concatenated,
       LENGTH(string_col) AS str_length
FROM type_test_table;

-- PROBLEM: CHAR type mostly works but LENGTH may be wrong for padded strings
SELECT char_col,
       char_col = 'fixed     ' AS exact_match,
       TRIM(char_col) AS trimmed,
       LENGTH(char_col) AS char_length
FROM type_test_table;

-- PROBLEM: Decimal/numeric arithmetic completely broken
-- 12345.67 + 98765.43210 = 1244443543 (wrong), should be ~111111.10
-- Decimal multiplication also produces garbage values
SELECT decimal_col,
       numeric_col,
       decimal_col > 0 AS decimal_positive,
       numeric_col < 50000 AS numeric_small,
       decimal_col + numeric_col AS sum_decimals,
       decimal_col * 2 AS decimal_doubled
FROM type_test_table;

-- PROBLEM: Decimal/numeric aggregations return wrong values
-- SUM should be much larger, not 17
-- Type handling appears completely broken
SELECT SUM(decimal_col) AS sum_decimal,
       SUM(numeric_col) AS sum_numeric,
       AVG(decimal_col) AS avg_decimal,
       AVG(numeric_col) AS avg_numeric,
       MIN(decimal_col) AS min_decimal,
       MAX(numeric_col) AS max_numeric
FROM type_test_table;

-- PROBLEM: Date comparisons may work but display format is wrong
SELECT date_col,
       date_col > '2024-06-01' AS after_june,
       date_col < '2024-12-01' AS before_december,
       date_col = '2024-02-29' AS is_leap_day
FROM type_test_table;

-- PROBLEM: Date aggregations return day counts from epoch instead of dates
-- MIN(date_col) returns 8780 (days), should be '2024-01-15'
-- MAX(date_col) returns 9131 (days), should be '2024-12-31'
SELECT MIN(date_col) AS earliest_date,
       MAX(date_col) AS latest_date,
       COUNT(DISTINCT date_col) AS unique_dates
FROM type_test_table;

-- PROBLEM: Timestamp comparisons all return wrong values
-- All timestamps incorrectly show after_june = t
-- Comparison logic appears broken
SELECT timestamp_col,
       timestamp_col > '2024-06-01 00:00:00' AS after_june,
       timestamp_col < '2024-12-01 00:00:00' AS before_december,
       timestamp_col = '2024-01-15 10:30:00' AS exact_match
FROM type_test_table;

-- PROBLEM: Timestamp aggregations return Unix seconds instead of timestamps
-- MIN returns 1136572928 (seconds), should be timestamp format
-- MAX returns 2127855615 (seconds), should be timestamp format
SELECT MIN(timestamp_col) AS earliest_timestamp,
       MAX(timestamp_col) AS latest_timestamp,
       COUNT(DISTINCT timestamp_col) AS unique_timestamps
FROM type_test_table;

-- PROBLEM: Interval comparisons mostly work but year comparison wrong
-- Row 3 with '1 year 2 months' shows less_than_year = f (wrong)
SELECT interval_col,
       interval_col > INTERVAL '1 day' AS more_than_day,
       interval_col < INTERVAL '1 year' AS less_than_year
FROM type_test_table;

-- PROBLEM: Type casting not going through MLIR (no NOTICE message)
-- Results may be PostgreSQL native, not MLIR compiled
SELECT int2_col::float4 > float4_col AS int_to_float_compare,
       int4_col + float4_col AS int_plus_float,
       decimal_col::float8 AS decimal_to_float
FROM type_test_table;

-- PROBLEM: COUNT returns 5 for all columns including NULLs
-- Should return 4 for non-NULL counts (one row has all NULLs)
SELECT COUNT(*) AS total_rows,
       COUNT(bool_col) AS non_null_bool,
       COUNT(int4_col) AS non_null_int,
       COUNT(float8_col) AS non_null_float,
       COUNT(string_col) AS non_null_string,
       COUNT(decimal_col) AS non_null_decimal,
       COUNT(date_col) AS non_null_date,
       COUNT(timestamp_col) AS non_null_timestamp,
       COUNT(interval_col) AS non_null_interval
FROM type_test_table;

-- PROBLEM: SELECT * shows correct data structure but some formatting issues
-- This query helps verify what data was actually inserted
SELECT * FROM type_test_table;

DROP TABLE type_test_table;
