#!/bin/sh
# Creates the database used by the integration test suite, so tests never write
# to development data. Runs once, on first cluster initialisation.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
    CREATE DATABASE ${POSTGRES_DB}_test;
SQL
