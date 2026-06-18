LOAD
'pgx_lower';

DROP TABLE IF EXISTS test_text;

CREATE TABLE test_text
(
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(50),
    description TEXT,
    category    VARCHAR(20)
);

INSERT INTO test_text(name, description, category)
VALUES ('Apple', 'Fresh red apple fruit', 'fruit'),
       ('Banana', 'Yellow curved banana', 'fruit'),
       ('Carrot', 'Orange root vegetable', 'vegetable'),
       ('Avocado', 'Green creamy avocado', 'fruit'),
       ('Spinach', 'Green leafy vegetable', 'vegetable');

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_13_text_operations_001 */
SELECT (name LIKE 'A%') AS starts_with_a
FROM test_text;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_13_text_operations_002 */
SELECT (name LIKE '%a%') AS contains_a
FROM test_text;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_13_text_operations_003 */
SELECT (description LIKE '%green%') AS contains_green
FROM test_text;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_13_text_operations_004 */
SELECT (category LIKE 'fruit') AS is_fruit
FROM test_text;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_13_text_operations_005 */
SELECT (name || ' - ' || description) AS concatenated
FROM test_text;
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_13_text_operations_006 */
SELECT (name || ' (' || category || ')') AS name_with_category
FROM test_text;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_13_text_operations_007 */
SELECT SUBSTRING(name FROM 1 FOR 3) AS name_prefix
FROM test_text;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_13_text_operations_008 */
SELECT SUBSTRING(description FROM 1 FOR 10) AS desc_start
FROM test_text;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_13_text_operations_009 */
SELECT UPPER(name) AS upper_name
FROM test_text;
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_13_text_operations_010 */
SELECT LOWER(description) AS lower_desc
FROM test_text;

DROP TABLE test_text;
