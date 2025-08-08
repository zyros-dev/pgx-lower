-- Test more complex operations that might trigger different crash behavior
SELECT * FROM test;
SELECT * FROM test WHERE id > 0;
SELECT id * 2 FROM test;
SELECT id + 1 FROM test WHERE id = 1;