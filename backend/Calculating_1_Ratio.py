import pandas as pd

df = pd.read_csv("labels2.csv")   # poora load hota hai, par sirf ek column ka count karega
print(df["label"].value_counts())   # kitne 0 aur 1
print("Proportion of 1s:", df["label"].mean())  # 1 ka ratio
