LOAD 'pgx_lower.so';
SET client_min_messages TO NOTICE;

-- Test selecting just char column
DROP TABLE IF EXISTS char_only;
CREATE TABLE char_only (
    ch CHAR(10)
);
INSERT INTO char_only VALUES (LPAD('ch1', 10, 'x'));
INSERT INTO char_only VALUES (LPAD('ch2', 10, 'x'));
INSERT INTO char_only VALUES (LPAD('ch3', 10, 'x'));
SELECT ch FROM char_only;