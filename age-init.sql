-- Runs once on first database initialization (mounted into
-- /docker-entrypoint-initdb.d). Creates both extensions the AGE driver needs.
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS vector;

-- Make ag_catalog part of the default search_path for this database so that
-- direct (non-pool) connections — psql, tests — can resolve cypher()/agtype
-- without a per-session SET. The driver's pool additionally sets it per-acquire.
ALTER DATABASE age_test SET search_path = ag_catalog, "$user", public;
