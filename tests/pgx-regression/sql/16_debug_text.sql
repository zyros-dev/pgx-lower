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

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_16_debug_text_001 */
SELECT ch
FROM char_only;