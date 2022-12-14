import os
import pandas as pd

total_path = "data/"
files_path = "data/tagtog_csv/"
eda_save_path = "data/EDA_results/"

files = []
for (root, directories, file) in os.walk(files_path):
    for f in file:
        if ".csv" in f:
            file_path = os.path.join(root, f)
            files.append(file_path)

df = pd.read_csv(files[0])
for i in range(1, len(files)):
    df = pd.concat([df, pd.read_csv(files[i])])

subjects = df["subject"].apply(lambda x: eval(x)["text"])
objects = df["object"].apply(lambda x: eval(x)["text"])

pairs = list(zip(subjects, objects))
df["pair"] = pairs
pair_dist = df["pair"].value_counts(sort=True, ascending=False)
label_dist = df["label"].value_counts()

pair_dist.to_csv(f"{eda_save_path}pair_dist.csv")
label_dist.to_csv(f"{eda_save_path}label_dist.csv")
df.to_csv(f"{files_path}total_data.csv", index=False)
