-- Test special operators: PgBetweenOp, PgInOp, PgCaseOp  
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_special;

CREATE TABLE test_special (
    id SERIAL PRIMARY KEY,
    value INTEGER,
    category INTEGER,       -- Replaces CHAR(1): 1='A', 2='B', 3='C'
    score INTEGER,        -- Replaces DECIMAL(5,2): scaled by 100 (85.50 = 8550)
    active BOOLEAN
);

-- Insert test data using only minimal supported types
INSERT INTO test_special(value, category, score, active) VALUES 
    (15, 1, 8550, true),        -- 'A' = 1, 85.50 = 8550
    (25, 2, 9275, false),       -- 'B' = 2, 92.75 = 9275
    (35, 1, 7825, true),        -- 'A' = 1, 78.25 = 7825
    (45, 3, 9500, true),        -- 'C' = 3, 95.00 = 9500
    (55, 2, 8880, false),       -- 'B' = 2, 88.80 = 8880
    (65, 1, 9120, true),        -- 'A' = 1, 91.20 = 9120
    (75, 3, 8240, false);       -- 'C' = 3, 82.40 = 8240

-- Test BETWEEN operations with integers
-- These should trigger MLIR compilation with BETWEEN operators
SELECT (value BETWEEN 20 AND 50) AS in_range_20_50 FROM test_special;
SELECT (value BETWEEN 30 AND 60) AS in_range_30_60 FROM test_special;
SELECT (score BETWEEN 8000 AND 9000) AS score_in_range FROM test_special;
SELECT (value NOT BETWEEN 40 AND 70) AS not_in_range FROM test_special;

-- Test IN operations with integers
-- These should trigger MLIR compilation with IN operators
SELECT (value IN (15, 25, 35)) AS in_low_values FROM test_special;
SELECT (value IN (45, 55, 65, 75)) AS in_high_values FROM test_special;
SELECT (category IN (1, 2)) AS in_categories_ab FROM test_special;
SELECT (value NOT IN (25, 45, 65)) AS not_in_specific FROM test_special;

-- Test CASE operations with integers
-- These should trigger MLIR compilation with CASE operators
SELECT CASE WHEN value < 30 THEN 1 WHEN value < 60 THEN 2 ELSE 3 END AS value_category FROM test_special;
SELECT CASE category WHEN 1 THEN 10 WHEN 2 THEN 20 ELSE 30 END AS category_name FROM test_special;
SELECT CASE WHEN active THEN 1 ELSE 0 END AS status FROM test_special;
SELECT CASE WHEN score > 9000 THEN 1 WHEN score > 8000 THEN 2 ELSE 3 END AS grade FROM test_special;

DROP TABLE test_special;