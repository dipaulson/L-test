import numpy as np
import pandas as pd

num_jobs = 2000
group_size = 10
total = num_jobs // group_size
signals = np.array([0.1, 0.2, 0.4, 0.6, 0.8])

def read_and_reindex(path):
    df = pd.read_csv(path, index_col=0)
    idx = df.index.astype(str).str.strip()
    sig = np.array([float(s.split("_")[1]) for s in idx])
    did = np.array([int(s.split("_")[2]) for s in idx])
    df.index = pd.MultiIndex.from_arrays([sig, did], names=["Signal", "Dataset"])
    return df

batch_results = []

for b in range(total):
    dfs = []
    for j in range(group_size):
        file_id = b * group_size + j + 1
        path = f"pvals_rep_{file_id}.csv"
        dfs.append(read_and_reindex(path))

    combined = pd.concat(dfs, axis=0)

    rows = []
    for s in signals:
        block = combined.loc[s]
        within = np.log(np.sqrt(np.mean(np.var(block.to_numpy(), axis=1, ddof=1)))).item()
        overall = np.log(np.std(block.to_numpy(), ddof=1)).item()
        rows.append((s, within, overall))

    batch_df = pd.DataFrame(rows, columns=["Signal", "Within", "Overall"]).set_index("Signal")
    batch_results.append(batch_df)

# Aggregate across the 200 batches
stack = np.stack([bdf.loc[signals, ["Within", "Overall"]].to_numpy() for bdf in batch_results], axis=0)
mean_vals = stack.mean(axis=0)                      
se_vals = stack.std(axis=0, ddof=0) / np.sqrt(total)

mean_df = pd.DataFrame(mean_vals, index=signals, columns=["Within", "Overall"]).reset_index().rename(columns={"index":"Signal"})
se_df   = pd.DataFrame(se_vals,   index=signals, columns=["Within", "Overall"]).reset_index().rename(columns={"index":"Signal"})

# Save dfs
mean_df.to_csv("means.csv")
se_df.to_csv("ses.csv")
