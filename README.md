# Data Annotation for RE Wrap Up

## 프로젝트 개요
본 프로젝트는 ‘대한민국 대통령’ 이라는 주제에서 도출된 키워드들을 위키피디아 문서 제목으로 설정하여 수집한 데이터를 관계 추출 테스크 데이터셋으로 제작하였습니다. <br/>
Relation set의 구성 및 정의, 가이드라인 작성, 파일럿 및 메인 어노테이션, Fleiss Kappa 계산, 모델 Fine-tuning의 과정을 진행하였습니다.
(한국어 위키피디아 CC BY-SA 3.0)

<br/>

## 팀 구성원

정준녕|임성근|이원재|이정아|김별희|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/51015187?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/14817039?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/61496071?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/65378914?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/42535803?v=4' height=80 width=80px></img>
[Github](https://github.com/ezez-refer)|[Github](https://github.com/lim4349)|[Github](https://github.com/wjlee-ling)|[Github](https://github.com/jjeongah)|[Github](https://github.com/kimbyeolhee)


<br>

* [Annotation Guideline](https://nextlevelpotato.notion.site/KLUE-RE-Annotation-guideline-c969709aaecf481bb580c913ef0446b1)
* [Wrap Up Report](https://nextlevelpotato.notion.site/Data-Annotation-for-RE-Wrap-Up-3122a61f46284e4f9180ca9c1270a29a)

<br/>
<br/>

---

## How to Use

### tagtog의 json file을 csv로 변환하는 법

```
@kimbyeolhee

💡 해당 문장 txt 파일과 tagtog으로 부터 annotation을 진행한 json 파일을 한 폴더에 넣어주세요(line 94) 💡
❗❗ Entity는 [SUBJ/OBJ]-[ENTITY TYPE]-[label] 형식으로 태깅이 되어있어야 합니다. ❗❗ 

1. tagtog_to_csv.py의 94 line의 files_path에 txt와 json 파일이 들어있는 경로 입력
2. tagtog_to_csv.py의 95 line에 csv들로 변환하여 저장할 경로 입력
3. 96 line에 annotator 입력

4. python tagtog_to_csv.py 실행
```


<br/>

### tagtog_csv 파일들의 라벨과 entity pair 확인하는 법

```
@ezez-refer

1. eda.py의 total_path에 tagtog의 json file을 csv로 변환한 파일들의 합본을 저장할 경로 설정
2. eda.py의 files_path에 tagtog의 json file을 csv로 변환한 파일들이 있는 폴더로 설정
3. eda.py의 eda_save_path에 EDA 결과를 저장할 경로를 설정

3. python eda.py 실행

```

<br/>

### 메인 태깅 후 annotator로 참여한 라벨을 불러오고 iaa를 위해 label을 num 값으로 변경하는 법

```
@ezez-refer

1. 56 line에 main tagging한 csv 파일 경로 설정
2. 57 line에 출력 결과물을 저장할 경로 및 이름 설정

3. python label_to_num.py 실행

```

### 코드 기준 파일 구조

```
├─📁data
| ├─📁EDA_results
| | ├─label_dist.csv # eda.py 실행 시 생성되는 파일
| | └─pair_dist.csv  # eda.py 실행 시 생성되는 파일
| ├─📁main_tagging_data
| | ├─NLP13-main_tagging_kbh.csv
| | └─NLP13-main_tagging_kbh.label_to_num.csv  # label_to_num 실행 시 생성되는 파일
| ├─📁raw_data
| | ├─정치.json
| | └─정치.txt
| ├─📁tagtog_csv
| | ├─total_data.csv # eda.py 실행 시 생성되는 파일
| | └─정치.csv  # tagtog_to_csv.py 실행 시 생성되는 파일
| ├─label.json
| └─total_data.csv
├─calculate_iaa.py
├─eda.py
├─fleiss.py
├─label_to_num.py
├─README.md
├─requirements.txt
└─tagtog_to_csv.py
```
