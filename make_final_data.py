import os
import pandas as pd

files_path = "data/main_tagging_data/"
save_path = "src/data/raw_data/"

num_label_to_en = {
    1 : "no_relation",
    2 : "org:founded_by",
    3 : "per:employee_of",
    4 : "per:colleagues",
    5 : "per:title",
    6 : "per:inauguration/removal_date",
    7 : "per:origin",
    8 : "loc:subordinate",
    9 : "event/artifact:date",
}


files = []
for (root, directories, file) in os.walk(files_path):
    for f in file:
        if "NLP13-main_tagging-label" in f:
            file_path = os.path.join(root, f)
            files.append(file_path)

# 과반수 라벨을 정답 라벨로 설정
main_df = pd.read_csv(files[0])
main_df = main_df[["id", "sentence", "subject_entity", "object_entity", "label_num"]]

for i in range(1, len(files)):
    labels = pd.read_csv(files[i])["label_num"]
    main_df[f"label_num_{i}"] = labels

final_labels = []
for i in range(len(main_df)):
    temp = main_df.iloc[i][4:]
    most_freq_label = temp.value_counts().index[0]
    final_labels.append(most_freq_label)

main_df["final_num_label"] = final_labels
main_df["label"] = main_df["final_num_label"].map(num_label_to_en)
main_df = main_df[["id", "sentence", "subject_entity", "object_entity", "label"]]

main_df.to_csv(f"{save_path}total_data.csv", index=False)

