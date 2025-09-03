LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS char_only;

CREATE TABLE char_only
(
    ch CHAR(10)
);

INSERT INTO char_only
VALUES (LPAD('ch1', 10, 'x'));
INSERT INTO char_only
VALUES (LPAD('ch2', 10, 'x'));
INSERT INTO char_only
VALUES (LPAD('ch3', 10, 'x'));

SELECT ch
FROM char_only;