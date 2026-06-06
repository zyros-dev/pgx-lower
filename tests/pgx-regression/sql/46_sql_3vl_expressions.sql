LOAD 'pgx_lower.so';

CREATE TEMP TABLE pgx_3vl_expr_source(f boolean, t boolean, n boolean);
INSERT INTO pgx_3vl_expr_source VALUES (false, true, NULL);

/* <<pgx-lower-config>>: auto_should_route_to=ignore id=pgx_46_sql_3vl_expressions_pset */
\pset format unaligned

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_46_sql_3vl_expressions_001 */
SELECT f AND n AS false_and_null,
       t OR n AS true_or_null,
       n AND t AS null_and_true,
       n OR f AS null_or_false
FROM pgx_3vl_expr_source;

CREATE TEMP TABLE pgx_agg_source(group_id int, amount numeric);
INSERT INTO pgx_agg_source VALUES (1, 10.0), (1, NULL), (2, NULL);

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_46_sql_3vl_expressions_002 */
SELECT count(*) AS count_all,
       count(amount) AS count_amount,
       sum(amount) IS NULL AS sum_no_rows_is_null
FROM pgx_agg_source
WHERE group_id = 9999;

/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_46_sql_3vl_expressions_003 */
SELECT group_id,
       count(*) AS count_all,
       count(amount) AS count_amount,
       sum(amount) IS NULL AS sum_is_null
FROM pgx_agg_source
GROUP BY group_id
ORDER BY group_id;

/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_46_sql_3vl_expressions_004 */
SELECT t IS TRUE AS t_is_true,
       n IS UNKNOWN AS n_is_unknown
FROM pgx_3vl_expr_source;

DROP TABLE pgx_agg_source;
DROP TABLE pgx_3vl_expr_source;
