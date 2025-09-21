import numpy as np
import pandas as pd
import glob

# Multiple testing procedures
def holm(p, alpha=0.05):
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    rejects = 0
    for i in range(m):
        threshold = alpha / (m - i)
        if sorted_p[i] <= threshold:
            rejects += 1
        else:
            break
    return rejects / m

def BHq(p, alpha=0.10):
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    cmp = sorted_p <= thresholds
    if not np.any(cmp):
        rejects = 0
    else:
        rejects = np.max(np.nonzero(cmp)[0]) + 1
    return rejects / m

def Bonf(p, alpha=0.05):
    m = p.size
    threshold = alpha / m
    return np.sum(p<=threshold) / m

# Create dataframe for test powers
MC_samples = 500 # Adjust based on how many MC samples used for L-test
tests = [('F', 0), ('L', 1), ('R', 2)] # Adjust based on which tests were run
avg_dfs = []
for i in range(1, 51):
    per_iter = []
    for j in range(1, 17):
        p_vals = pd.read_csv(f"p_vals_reg_{j}_it_{i}.csv", header=None).to_numpy()
        rows = []
        for name, col in tests:
            pcol = p_vals[:, col].astype(float)
            holm_prop = holm(pcol)
            bonf_prop = Bonf(pcol)
            bhq_prop  = BHq(pcol)
            rows.append((name, holm_prop, bonf_prop, bhq_prop))
        df_j = pd.DataFrame(rows, columns=['Test', 'Holm (5%)', 'Bonf (5%)', 'BHq (10%)']).set_index('Test')
        per_iter.append(df_j)
    avg_df = (
        pd.concat(per_iter, keys=range(1, 17), names=['reg', 'test'])
          .groupby(level='test')
          .mean()
    )
    avg_df.index.name = "test"
    avg_dfs.append(avg_df)

stacked = pd.concat(avg_dfs, keys=range(1, 51), names=['iter', 'test'])
mean_df = stacked.groupby(level='test').mean()
std_df  = stacked.groupby(level='test').std(ddof=0)
count_df = stacked.groupby(level='test').count()
se_df = std_df / np.sqrt(count_df)

out_cols = []
for col in mean_df.columns:
    out_cols.append(mean_df[col].rename(f"{col} (avg)"))
    out_cols.append(se_df[col].rename(f"{col} (se)"))

powers_summary = pd.concat(out_cols, axis=1)
powers_summary.index.name = "Test"
powers_summary.to_csv(f"powers_MC={MC_samples}.csv", float_format="%.18e")


# Create dataframe for test times
file_list = glob.glob("times_reg_*_it_*.csv")
all_times = []
for f in file_list:
    df = pd.read_csv(
        f,
        header=None,
        usecols=[0, 1, 2],              
        names=["F", "L", "R"]          
    ) # Adjust based on which tests were run 
    all_times.append(df)
times_concat = pd.concat(all_times, ignore_index=True)
times_mean = times_concat.mean(numeric_only=True)
times_std  = times_concat.std(ddof=0, numeric_only=True)
times_count = times_concat.count(numeric_only=True)
times_se = times_std / np.sqrt(times_count)
times_summary = (
        pd.concat([times_mean.rename("Avg"), times_se.rename("SE")], axis=1)
        .reset_index()
        .rename(columns={"index": "Test"})
    )
times_summary.to_csv(f"times_MC={MC_samples}.csv", index=False, float_format="%.18e")