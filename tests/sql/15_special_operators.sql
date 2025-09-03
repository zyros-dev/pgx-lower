LOAD
'pgx_lower.so';

DROP TABLE IF EXISTS test_special;

CREATE TABLE test_special
(
    id       SERIAL PRIMARY KEY,
    value    INTEGER,
    category INTEGER,
    score    INTEGER,
    active   BOOLEAN
);

INSERT INTO test_special(value, category, score, active)
VALUES (15, 1, 8550, true),
       (25, 2, 9275, false),
       (35, 1, 7825, true),
       (45, 3, 9500, true),
       (55, 2, 8880, false),
       (65, 1, 9120, true),
       (75, 3, 8240, false);

SELECT (value BETWEEN 20 AND 50) AS in_range_20_50
FROM test_special;
SELECT (value BETWEEN 30 AND 60) AS in_range_30_60
FROM test_special;
SELECT (score BETWEEN 8000 AND 9000) AS score_in_range
FROM test_special;
SELECT (value NOT BETWEEN 40 AND 70) AS not_in_range
FROM test_special;

SELECT (value IN (15, 25, 35)) AS in_low_values
FROM test_special;
SELECT (value IN (45, 55, 65, 75)) AS in_high_values
FROM test_special;
SELECT (category IN (1, 2)) AS in_categories_ab
FROM test_special;
SELECT (value NOT IN (25, 45, 65)) AS not_in_specific
FROM test_special;

SELECT CASE WHEN value < 30 THEN 1 WHEN value < 60 THEN 2 ELSE 3 END AS value_category
FROM test_special;
SELECT CASE category WHEN 1 THEN 10 WHEN 2 THEN 20 ELSE 30 END AS category_name
FROM test_special;
SELECT CASE WHEN active THEN 1 ELSE 0 END AS status
FROM test_special;
SELECT CASE WHEN score > 9000 THEN 1 WHEN score > 8000 THEN 2 ELSE 3 END AS grade
FROM test_special;

DROP TABLE test_special;