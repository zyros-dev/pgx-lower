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

-- Enable debug logging for LIKE operator debugging
SET client_min_messages TO DEBUG1;
SET pgx_lower.log_enable = true;
SET pgx_lower.log_debug = true;
SET pgx_lower.log_trace = true;
SET pgx_lower.enabled_categories = 'AST_TRANSLATE,RELALG_LOWER,DB_LOWER,RUNTIME';

SELECT (name LIKE 'A%') AS starts_with_a
FROM test_text;

-- Disable debug logging
SET client_min_messages TO NOTICE;
SET pgx_lower.log_enable = false;
-- Temporarily commenting out other operations to isolate the issue
-- SELECT (name LIKE '%a%') AS contains_a
-- FROM test_text;
-- SELECT (description LIKE '%green%') AS contains_green
-- FROM test_text;
-- SELECT (category LIKE 'fruit') AS is_fruit
-- FROM test_text;
-- SELECT (name || ' - ' || description) AS concatenated
-- FROM test_text;
-- SELECT (name || ' (' || category || ')') AS name_with_category
-- FROM test_text;
-- SELECT SUBSTRING(name FROM 1 FOR 3) AS name_prefix
-- FROM test_text;
-- SELECT SUBSTRING(description FROM 1 FOR 10) AS desc_start
-- FROM test_text;
-- SELECT UPPER(name) AS upper_name
-- FROM test_text;
-- SELECT LOWER(description) AS lower_desc
-- FROM test_text;

DROP TABLE test_text;