-- Test basic extension loading
CREATE EXTENSION IF NOT EXISTS pgx_lower;

-- Test that extension is loaded
SELECT count(*) FROM pg_extension WHERE extname = 'pgx_lower';

-- Test basic functionality
SELECT try_cpp_executor(NULL); 