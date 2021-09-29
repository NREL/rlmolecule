#!/bin/bash
if [ `hostname` != $DB_HOST ]; then
   exit 0
fi
echo "Running postgres on `hostname`"
module load conda
conda activate /scratch/hsorense/conda/rlmolecule
initdb -D qed_data_psql.db
# Add all hosts to pg_hba.conf
for h in `scontrol show hostnames`; do
    ip=`host $h | awk '{print $(NF)}'`
    echo "host    all             all             $ip/32            trust">>qed_data_psql.db/pg_hba.conf
done

pg_ctl -D qed_data_psql.db -l postgres.log -o "-i" start
psql -c "CREATE USER example_user WITH PASSWORD 'tmppassword'" postgres
echo "tmppassword" > psql_pass
createdb --owner=example_user bde
exit 0 # In case there are errors with the database calls
