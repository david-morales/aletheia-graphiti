-- Runs once on first database initialization (mounted into
-- /docker-entrypoint-initdb.d). Creates both extensions the AGE driver needs.
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS vector;
