#!/bin/bash

initdb -D psql_data.db
pg_ctl -D psql_data.db -l postgres.log -o "-i" start
psql -c "CREATE USER example_user WITH PASSWORD 'tmppassword'" postgres
createdb --owner=example_user rl
exit 0 # In case there are errors with the database calls


# stop with pg_ctl -D qed_data_psql.db stop