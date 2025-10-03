-- Copyright 2020 Mats Kindahl
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

LOAD 'pgx_lower.so';

DO $$ BEGIN
    RAISE NOTICE 'MLIR JIT Engine extension installed successfully.';
END $$;

CREATE FUNCTION pgx_lower_test_relalg(text) RETURNS SETOF record
    AS 'pgx_lower.so', 'pgx_lower_test_relalg'
    LANGUAGE C STRICT;
