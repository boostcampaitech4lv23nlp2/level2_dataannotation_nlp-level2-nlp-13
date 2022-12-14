import os
import json
import pandas as pd


def get_labels(f_path="./data/label.json"):
    with open(f_path, "r") as f:
        labels = json.load(f)

    return labels


def tagtog_to_csv(file_name, txt_file_path, json_file_path):
    # txt 파일의 문장 로드
    sentences = ""
    with open(txt_file_path, "r") as f:
        while True:
            if text := f.readline():
                sentences += text
            else:
                break

    # json 파일 로드
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    json_file = eval(str(json_file))

    # 매칭할 label 로드
    labels = get_labels()

    # 각 문장 별 entity 정보 및 라벨 추출
    id_list = []
    sentence_list = []
    subject_entity_list = []
    object_entity_list = []
    label_list = []

    relations = json_file["relations"]
    for sent_id, relation in enumerate(relations):
        subj_type, obj_type, label = labels[relation["classId"]].split("-")

        # entity 정보 추출
        entity1 = relation["entities"][0].split("|")[1]
        entity1_is_subject = labels[entity1].split("-")[1] == "SUBJ"
        if entity1_is_subject:
            subj_idxes = relation["entities"][0].split("|")[-1].split(",")
        else:
            obj_idxes = relation["entities"][0].split("|")[-1].split(",")

        entity2 = relation["entities"][1].split("|")[1]
        entity2_is_subject = labels[entity2].split("-")[1] == "SUBJ"
        if entity2_is_subject:
            subj_idxes = relation["entities"][1].split("|")[-1].split(",")
        else:
            obj_idxes = relation["entities"][1].split("|")[-1].split(",")

        subj_word = sentences[int(subj_idxes[0]) : int(subj_idxes[1]) + 1]
        obj_word = sentences[int(obj_idxes[0]) : int(obj_idxes[1]) + 1]

        # 해당 문장 추출
        start_n_idx = sentences.rfind("\n", 0, int(subj_idxes[0]))
        end_n_idx = sentences.find("\n", int(subj_idxes[0]))
        sentence = sentences[start_n_idx + 1 : end_n_idx]

        # 해당 문장에 맞는 entity idx 추출
        sent_subj_word_start_idx = int(subj_idxes[0]) - start_n_idx - 1
        sent_subj_word_end_idx = int(subj_idxes[1]) - start_n_idx - 1
        sent_obj_word_start_idx = int(obj_idxes[0]) - start_n_idx - 1
        sent_obj_word_end_idx = int(obj_idxes[1]) - start_n_idx - 1

        subject_info = {
            "word": subj_word,
            "start_idx": sent_subj_word_start_idx,
            "end_idx": sent_subj_word_end_idx,
            "entity_type": subj_type,
        }
        object_info = {
            "word": obj_word,
            "start_idx": sent_obj_word_start_idx,
            "end_idx": sent_obj_word_end_idx,
            "entity_type": obj_type,
        }

        id_list.append(f"{file_name}_{sent_id}")
        sentence_list.append(sentence)
        label_list.append(label)
        subject_entity_list.append(subject_info)
        object_entity_list.append(object_info)

    return id_list, sentence_list, subject_entity_list, object_entity_list, label_list


if __name__ == "__main__":
    files_path = "data/raw_data/"
    saved_path = "data/tagtog_csv/"
    annotator = "김별희"

    json_files = []
    txt_files = []
    for (root, directories, file_names) in os.walk(files_path):
        for file in file_names:
            if file.endswith(".json"):
                json_files.append(file)
            elif file.endswith(".txt"):
                txt_files.append(file)

    json_files = sorted(json_files)
    txt_files = sorted(txt_files)

    for json_file, txt_file in zip(json_files, txt_files):
        json_file_name = json_file.split(".")[0]
        txt_file_name = txt_file.split(".")[0]
        if json_file_name != txt_file_name:
            print(f"{json_file_name} and {txt_file_name} are not matched")
            break
        else:
            (
                id_list,
                sentence_list,
                subject_entity_list,
                object_entity_list,
                label_list,
            ) = tagtog_to_csv(
                json_file_name, f"{files_path}{txt_file}", f"{files_path}{json_file}"
            )
            df = {
                "id": id_list,
                "sentence": sentence_list,
                "subject_entity": subject_entity_list,
                "object_entity": object_entity_list,
                "label": label_list,
                "annotator": annotator,
            }
            df = pd.DataFrame(df)
            df.to_csv(f"{saved_path}{json_file_name}.csv", index=False)
