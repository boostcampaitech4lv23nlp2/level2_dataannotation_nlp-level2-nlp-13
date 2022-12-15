import os
from collections import defaultdict
from copy import deepcopy
from itertools import cycle

import pandas as pd

files_path = "data/main_tagging_data/"
save_path = "src/data/raw_data/"

num_label_to_en = {
    1: "no_relation",
    2: "org:founded_by",
    3: "per:employee_of",
    4: "per:colleagues",
    5: "per:title",
    6: "per:inauguration/removal_date",
    7: "per:origin",
    8: "loc:subordinate",
    9: "event/artifact:date",
}


files = []
for (root, directories, file) in os.walk(files_path):
    for f in file:
        if "NLP13-main_tagging-label" in f:
            file_path = os.path.join(root, f)
            files.append(file_path)
print(files)
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

# 중복 3번 초과일 경우 3개까지만 사용
del_idx = []  # 삭제할 인덱스를 담은 리스트
# 우선 중복인 temp_df 생성
temp_df = main_df.loc[main_df.duplicated(subset=["sentence"])]
# 중복인 temp_df에서 3개 초과인 것만 추출
over_3_sents = temp_df["sentence"].value_counts()[temp_df["sentence"].value_counts() > 3].index
for sent in over_3_sents:
    over_3_idx = []  # 동일한 문장을 갖고 있는 데이터 인덱스
    temp = temp_df.loc[temp_df["sentence"] == sent]
    over_3_idx.extend(temp.id)

    # 각 문장의 entity type 확인
    sent_entity_type_dict = {}
    entity_type_cnt_dict = defaultdict(int)
    for idx in over_3_idx:
        sub_entity_type = eval(temp_df[temp_df["id"] == idx]["subject_entity"].values[0])["entity_type"]
        obj_entity_type = eval(temp_df[temp_df["id"] == idx]["object_entity"].values[0])["entity_type"]
        sub_obj_entity_type = "/".join([sub_entity_type, obj_entity_type])

        sent_entity_type_dict[idx] = sub_obj_entity_type  # {'한반도_29': 'LOC/LOC'}
        entity_type_cnt_dict[sub_obj_entity_type] += 1  # {'PER/DAT': 3}

    # 다행히 Entity type 종류가 4개인 경우가 없음
    # entity type 별로 1개 씩만 남기기 (만약 종류가 2종류라면 첫번째꺼에서 1개 더 가져오기)
    descent_entity_type_cnt_key = cycle(sorted(entity_type_cnt_dict, key=lambda x: entity_type_cnt_dict[x], reverse=True))
    print(descent_entity_type_cnt_key)

    for _ in range(3):
        value = next(descent_entity_type_cnt_key)
        a = [k for k, v in sent_entity_type_dict.items() if v == value][0]

        del sent_entity_type_dict[a]  # 포함할 데이터는 dict 목록에서 삭제

    # 삭제할 데이터 인덱스 추가
    del_idx.extend(list(sent_entity_type_dict.keys()))

# 삭제할 인덱스를 가진 데이터 삭제
main_df = main_df[~main_df["id"].isin(del_idx)]

main_df.to_csv(f"{save_path}total_data_ex.csv", index=False)
