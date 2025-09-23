//using baseline
import pandas as pd

df = pd.read_csv("labels.csv")
# majority class decide karein
majority = int(df["label"].mean() >= 0.05)   # returns 0 if <0.5 else 1
df["pred"] = majority
df.to_csv("labels_with_pred.csv", index=False)
print(f"Using majority class {majority} as baseline.")
