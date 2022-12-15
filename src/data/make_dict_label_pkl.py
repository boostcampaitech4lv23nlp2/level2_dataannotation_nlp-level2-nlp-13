label_to_num_dict = {
    "no_relation": 0,
    "org:founded_by": 1,
    "per:employee_of": 2,
    "per:colleagues": 3,
    "per:title": 4,
    "per:inauguration/removal_date": 5,
    "per:origin": 6,
    "loc:subordinate": 7,
    "event/artifact:date": 8,
}

num_to_label_dict = {
    0: "no_relation",
    1: "org:founded_by",
    2: "per:employee_of",
    3: "per:colleagues",
    4: "per:title",
    5: "per:inauguration/removal_date",
    6: "per:origin",
    7: "loc:subordinate",
    8: "event/artifact:date",
}


import pickle

with open("./src/data/dict_label_to_num.pkl", "wb") as f:
    pickle.dump(label_to_num_dict, f)

with open("./src/data/dict_num_to_label.pkl", "wb") as f:
    pickle.dump(num_to_label_dict, f)