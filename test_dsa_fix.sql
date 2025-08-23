-- Test DSA char append fix
SET pgx_lower.enabled = on;
SET pgx_lower.force_jit = on;
SET client_min_messages = DEBUG;

-- Simple string constant test
SELECT 'hello' AS test_string;