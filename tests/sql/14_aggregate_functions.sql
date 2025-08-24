-- Test aggregate functions: PgSumOp, PgCountOp, PgAvgOp, PgMinOp, PgMaxOp
LOAD 'pgx_lower.so';
-- CRITICAL: Do not change logging level - required for MLIR pipeline visibility
-- NOTICE level enables full debugging of PostgreSQL AST → RelAlg → DB → LLVM → JIT pipeline
-- WARNING level suppresses essential MLIR compilation logs and breaks debugging capability
SET client_min_messages TO NOTICE;

DROP TABLE IF EXISTS test_aggregates;

CREATE TABLE test_aggregates (
    id SERIAL PRIMARY KEY,
    category INTEGER,     -- Replaces VARCHAR(20): 1='A', 2='B', 3='C'
    amount INTEGER,      -- Replaces DECIMAL(10,2): stored as cents
    quantity INTEGER,
    price INTEGER        -- Replaces REAL: stored as cents
);

-- Insert test data using only minimal supported types
INSERT INTO test_aggregates(category, amount, quantity, price) VALUES 
    (1, 10050, 5, 2010),      -- 'A', $100.50, 5, $20.10
    (2, 20075, 3, 6692),      -- 'B', $200.75, 3, $66.92
    (1, 15025, 7, 2146),      -- 'A', $150.25, 7, $21.46
    (3, 30000, 2, 15000),     -- 'C', $300.00, 2, $150.00
    (2, 17580, 4, 4395),      -- 'B', $175.80, 4, $43.95
    (1, 12560, 6, 2093),      -- 'A', $125.60, 6, $20.93
    (3, 25040, 1, 25040);     -- 'C', $250.40, 1, $250.40

-- Test aggregate functions without GROUP BY (whole table aggregates)
-- These should trigger MLIR compilation with aggregate operators
SELECT SUM(amount) AS total_amount_all FROM test_aggregates;
SELECT COUNT(*) AS total_rows FROM test_aggregates;
SELECT COUNT(DISTINCT category) AS unique_categories FROM test_aggregates;
SELECT AVG(amount) AS avg_amount_all FROM test_aggregates;
SELECT MIN(amount) AS min_amount_all FROM test_aggregates;
SELECT MAX(amount) AS max_amount_all FROM test_aggregates;
SELECT SUM(quantity) AS total_quantity FROM test_aggregates;
SELECT AVG(price) AS avg_price FROM test_aggregates;

DROP TABLE test_aggregates;