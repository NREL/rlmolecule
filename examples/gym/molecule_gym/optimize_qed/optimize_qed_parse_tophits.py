"""
Parses the <slurm-file>.top_hits file and writes a dataframe to 
<slurm-file>.top_hits.csv
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

results = {}
with open(args.filename, "r") as f:
    for line in f:
        qed = float(line.split("GraphGymEnv: ")[-1].split()[0])
        smiles = str(line.split("'smiles': '")[-1].split("'")[0]).strip()
        results[smiles] = qed

results = [(v, k) for k,v in results.items()]
df = pd.DataFrame(results, columns=["qed", "smiles"])
df = df.sort_values("qed", ascending=False)
df.to_csv(f"{args.filename}.csv")
print(df.head(20))
