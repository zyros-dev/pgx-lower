-- Test memory safety fixes for pgx-lower
-- Run with: psql -U postgres -d postgres -f test_memory_safety.sql

LOAD 'pgx_lower.so';
SET client_min_messages TO WARNING;

-- Test 1: Simple SELECT
DROP TABLE IF EXISTS test1;
CREATE TABLE test1(id INT);
INSERT INTO test1 VALUES (42);
SELECT * FROM test1;

-- Test 2: Multiple rows
DROP TABLE IF EXISTS test2;
CREATE TABLE test2(id INT);
INSERT INTO test2 VALUES (1), (2), (3);
SELECT * FROM test2;

-- Test 3: SERIAL type (like Test 1)
DROP TABLE IF EXISTS test3;
CREATE TABLE test3(id SERIAL);
INSERT INTO test3(id) VALUES (42);
SELECT * FROM test3;

-- Test 4: Multiple columns
DROP TABLE IF EXISTS test4;
CREATE TABLE test4(id INT, value INT);
INSERT INTO test4 VALUES (1, 100), (2, 200);
SELECT * FROM test4;

-- Clean up
DROP TABLE test1, test2, test3, test4;

SELECT 'All tests completed successfully!' AS result;