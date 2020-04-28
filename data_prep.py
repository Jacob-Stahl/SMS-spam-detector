import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv("spam.csv"))
df.drop(df.columns[[2,3,4]], axis=1, inplace=True)
df.rename(columns={"v1": "label", "v2": "text"}, inplace = True)

print(df)