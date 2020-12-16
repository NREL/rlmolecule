import psycopg2
import time
import logging
import random
import subprocess
import socket

dbparams = {
    'dbname': 'bde',
    'port': 5432,
    'host': 'yuma.hpc.nrel.gov',
    'user': 'bdeops',
    'password': '<removed>',
    'options': f'-c search_path=bde',
}


from bde.gaussian_redox import GaussianRedoxRunner


def run_optimization():

    with psycopg2.connect(**dbparams) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            WITH cte AS (
            SELECT id, smiles, mol_initial, estate
            FROM redoxcompound
            WHERE status = 'not started'
            ORDER BY id
            LIMIT 1
            FOR UPDATE
            )

            UPDATE redoxcompound SET status = 'in progress',
                                queued_at = CURRENT_TIMESTAMP,
                                node = %s
            FROM cte
            WHERE redoxcompound.id = cte.id
            RETURNING redoxcompound.id, redoxcompound.smiles, redoxcompound.mol_initial, redoxcompound.estate;
            """, (socket.gethostname(),))

            cid, smiles, mol_initial, estate = cur.fetchone()
            
    conn.close()

    try:
        
        runner = GaussianRedoxRunner(smiles, cid, mol_initial, estate, scratchdir='/scratch/pstjohn/')
        molstr, enthalpy, freeenergy, scfenergy, log = runner.process()
        
        with psycopg2.connect(**dbparams) as conn:
            with conn.cursor() as cur:   
                cur.execute("""
                UPDATE redoxcompound
                SET status = 'finished', 
                    mol_final = %s, enthalpy = %s, 
                    freeenergy = %s,
                    run_at = CURRENT_TIMESTAMP,
                    logfile = %s
                WHERE id = %s;""", 
                (molstr, enthalpy, freeenergy, log, cid))

        conn.close()


    except Exception as ex:

        with psycopg2.connect(**dbparams) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                UPDATE redoxcompound
                SET status = 'error', 
                    error = %s, 
                    run_at = CURRENT_TIMESTAMP
                WHERE id = %s;""", (str(ex), cid))  
        
        conn.close()
                
    return cid
                

if __name__ == "__main__":
    
    start_time = time.time()

    # Add a random delay to avoid race conditions at the start of the job
    time.sleep(random.uniform(0, 1*10))
    
    #while (time.time() - start_time) < (86400 * 1.75):  # Time in days
    while (time.time() - start_time) < (3600 * 3.5):  # Time in hours
        
        try:
            run_optimization()
            
        except psycopg2.OperationalError:
            time.sleep(5 + random.uniform(0, 60))
