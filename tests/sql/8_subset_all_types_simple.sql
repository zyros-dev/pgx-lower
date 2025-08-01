LOAD 'pgx_lower.so';
SET client_min_messages TO WARNING;
DROP TABLE IF EXISTS text_test;

CREATE TABLE text_test (
    char_col CHAR(10),
    varchar_col VARCHAR(255),
    text_col TEXT
);

INSERT INTO text_test VALUES ('hello', 'world', 'test');
SELECT char_col, varchar_col, text_col FROM text_test;