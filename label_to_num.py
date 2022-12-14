import pandas as pd

ko_label_to_en = {
    "관계_없음": "no_relation",
    "단체:창립자": "org:founded_by",
    "인물:소속단체": "per:employee_of",
    "인물:동료": "per:colleagues",
    "인물:직업/직함": "per:title",
    "인물:취임/퇴임일시": "per:inauguration/removal_date",
    "인물:국적/출신지": "per:origin",
    "지명:포함되는지역": "loc:subordinate",
    "사건/사물:일시": "event/artifact:date",
}

ko_label_to_num = {
    "관계_없음": 1,
    "단체:창립자": 2,
    "인물:소속단체": 3,
    "인물:동료": 4,
    "인물:직업/직함": 5,
    "인물:취임/퇴임일시": 6,
    "인물:국적/출신지": 7,
    "지명:포함되는지역": 8,
    "사건/사물:일시": 9,
}
en_label_to_num = {
    "no_relation": 1,
    "org:founded_by": 2,
    "per:employee_of": 3,
    "per:colleagues": 4,
    "per:title": 5,
    "per:inauguration/removal_date": 6,
    "per:origin": 7,
    "loc:subordinate": 8,
    "event/artifact:date": 9,
}


def fill_label(df):
    if df.label_ == "empty":
        df.label_ = df.label
    else:
        df.label_ = ko_label_to_en[df.label_]
    return df


def label_to_num(df):
    if df.label_ in ko_label_to_num:
        df["label_num"] = ko_label_to_num[df.label_]
    elif df.label_ in en_label_to_num:
        df["label_num"] = en_label_to_num[df.label_]
    return df


# csv 파일 로드
annotator = "lwj"
main_tagging_path = f"./data/main_tagging_data/NLP13-main_tagging_{annotator}.csv"
save_file_path = (
    f"./data/main_tagging_data/NLP13-main_tagging_{annotator}.label_to_num.csv"
)

df = pd.read_csv(main_tagging_path)
# df = df.drop("Unnamed: 8", axis=1)  # 불필요한 컬럼 삭제

# label_ 컬럼에 값이 없는 경우 label 컬럼의 값을 복사
df.label_ = df.label_.fillna("empty")
df = df.apply(fill_label, axis=1)
df = df.apply(label_to_num, axis=1)

# 결과 저장
df.to_csv(
    save_file_path,
    encoding="utf-8-sig",
)
