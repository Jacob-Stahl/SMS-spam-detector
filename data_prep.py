import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv("spam.csv"))
print(df)
df.drop(["job_id", "telecommuting", "has_company_logo", "has_questions", "employment_type", "required_experience", "required_education", "title", "location", "department", "salary_range", "industry", "function"], axis = 1, inplace = True)

raw_text = pd.Series([""] * len(df), index = df.index)
for i in range(len(df)): # combining raw text
    raw_text[i] = str(df["requirements"][i]) + " " + str(df["company_profile"][i]) + " " + str(df["description"][i]) + " " + str(df["benefits"][i]) + " "

df.drop(["requirements", "company_profile", "description", "benefits"], axis = 1, inplace = True)
df["raw_text"] = raw_text
print(df)
df.to_csv("prep_data.csv")