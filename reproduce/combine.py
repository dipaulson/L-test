import pandas as pd

num_sims = 1000
num_jobs = 250

powers = pd.read_csv("rejections_1.csv")

for i in range(2, num_jobs+1):
    df = pd.read_csv(f"rejections_{i}.csv")
    powers.iloc[:, 1:] += df.iloc[:, 1:]

powers.iloc[:, 1:] /= num_sims

powers.to_csv("avg_powers.csv", index=False)
