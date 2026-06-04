LOAD 'pgx_lower.so';

CREATE TABLE route_notice_smoke(id int4);
INSERT INTO route_notice_smoke VALUES (1), (2);

SET pgx_lower.execution_mode = 'force_fallback';
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_45_route_notices_001 */
SELECT id FROM route_notice_smoke ORDER BY id;

SET pgx_lower.execution_mode = 'auto';
/* <<pgx-lower-config>>: auto_should_route_to=lower id=pgx_45_route_notices_002 */
SELECT id FROM route_notice_smoke ORDER BY id;

SET pgx_lower.execution_mode = 'auto';
/* <<pgx-lower-config>>: auto_should_route_to=fallback id=pgx_45_route_notices_003 */
SELECT generate_series(1, 2);

SET pgx_lower.execution_mode = 'bogus';

DROP TABLE route_notice_smoke;
