LOAD 'pgx_lower.so';

DROP TABLE IF EXISTS simple_test;

CREATE TABLE simple_test
(
    id          INTEGER,
    col_a       INTEGER,
    col_b       INTEGER,
    col_c       INTEGER
);

INSERT INTO simple_test VALUES 
(1, 10, 20, 30),
(2, 11, 21, 31),
(3, 12, 22, 32);

-- Test basic column subset selection
-- Should show only col_b and col_c, not all columns
SELECT col_b, col_c FROM simple_test;

-- Test single column selection  
-- Should show only col_a
SELECT col_a FROM simple_test;

-- Test column reordering
-- Should show col_c, then col_a (reverse order)
SELECT col_c, col_a FROM simple_test;