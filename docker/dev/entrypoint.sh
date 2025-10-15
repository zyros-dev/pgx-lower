#!/bin/bash
set -e

# Start PostgreSQL as postgres user
su - postgres -c "/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/logfile start"

echo "PostgreSQL started"
echo "Dev container ready - workspace mounted at /workspace"

exec "$@"
