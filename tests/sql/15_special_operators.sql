-- Test special operators: PgBetweenOp, PgInOp, PgCaseOp
LOAD 'pgx_lower';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST  RelAlg  DB  LLVM  JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_special;

CREATE TABLE test_special (
    id SERIAL PRIMARY KEY,
    value INTEGER,
    category CHAR(1),
    score DECIMAL(5,2),
    active BOOLEAN
);

INSERT INTO test_special(value, category, score, active) VALUES 
    (15, 'A', 85.50, true),
    (25, 'B', 92.75, false),
    (35, 'A', 78.25, true),
    (45, 'C', 95.00, true),
    (55, 'B', 88.80, false),
    (65, 'A', 91.20, true),
    (75, 'C', 82.40, false);

-- Test BETWEEN operations in SELECT clauses
-- These should trigger MLIR compilation with BETWEEN operators
SELECT (value BETWEEN 20 AND 50) AS in_range_20_50 FROM test_special;
SELECT (value BETWEEN 30 AND 60) AS in_range_30_60 FROM test_special;
SELECT (score BETWEEN 80.0 AND 90.0) AS score_in_range FROM test_special;
SELECT (value NOT BETWEEN 40 AND 70) AS not_in_range FROM test_special;

-- Test IN operations in SELECT clauses
-- These should trigger MLIR compilation with IN operators
SELECT (value IN (15, 25, 35)) AS in_low_values FROM test_special;
SELECT (value IN (45, 55, 65, 75)) AS in_high_values FROM test_special;
SELECT (category IN ('A', 'B')) AS in_categories_ab FROM test_special;
SELECT (value NOT IN (25, 45, 65)) AS not_in_specific FROM test_special;

-- Test CASE operations in SELECT clauses
-- These should trigger MLIR compilation with CASE operators
SELECT CASE WHEN value < 30 THEN 'Low' WHEN value < 60 THEN 'Medium' ELSE 'High' END AS value_category FROM test_special;
SELECT CASE category WHEN 'A' THEN 'Alpha' WHEN 'B' THEN 'Beta' ELSE 'Gamma' END AS category_name FROM test_special;
SELECT CASE WHEN active THEN 'Active' ELSE 'Inactive' END AS status FROM test_special;
SELECT CASE WHEN score > 90 THEN 'Excellent' WHEN score > 80 THEN 'Good' ELSE 'Fair' END AS grade FROM test_special;

DROP TABLE test_special;