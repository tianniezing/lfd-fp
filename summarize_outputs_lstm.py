import re
import glob
import pandas as pd

results = []

# Loop through all relevant files
for file in glob.glob("output/slurm-*.out"):  # adjust folder/pattern if needed
    with open(file) as f:
        text = f.read()
        
        # Extract parameters from the new string format
        job_match = re.search(
            r"Job \d+: LR=([\d.]+) OPT=(\w+) BS=(\d+) LOSS=([\w_]+) DO=([\d.]+) RDO=([\d.]+) UNITS=(\d+) LAYERS=(\d+) BIDIR=(\w+)", 
            text
        )
        acc_match = re.search(r"Accuracy on own dev set: ([\d.]+)", text)
        
        if job_match and acc_match:
            lr = float(job_match.group(1))
            opt = job_match.group(2)
            bs = int(job_match.group(3))
            loss = job_match.group(4)
            dropout = float(job_match.group(5))
            rdo = float(job_match.group(6))
            units = int(job_match.group(7))
            layers = int(job_match.group(8))
            bidir = job_match.group(9) == "True"
            acc = float(acc_match.group(1))
            
            results.append({
                "file": file,
                "learning_rate": lr,
                "optimizer": opt,
                "batch_size": bs,
                "loss": loss,
                "dropout": dropout,
                "recurrent_dropout": rdo,
                "units": units,
                "layers": layers,
                "bidirectional": bidir,
                "accuracy": acc
            })

# Convert to DataFrame and sort by accuracy descending
df = pd.DataFrame(results)
df = df.sort_values(by="accuracy", ascending=False)

# Save to CSV
df.to_csv("summary.csv", index=False)

print(df)
