import json

from tqdm import tqdm

train_json = 'D:\dataset\data_pretrain/qa/avqa-train.json'
val_json = 'D:\dataset\data_pretrain/qa/avqa-val.json'
test_json = 'D:\dataset\data_pretrain/qa/avqa-test.json'

train_data = json.load(open(train_json, 'r', encoding='utf-8'))
val_data = json.load(open(val_json, 'r', encoding='utf-8'))
test_data = json.load(open(test_json, 'r', encoding='utf-8'))
all_data = train_data + val_data + test_data

# 统计答案的种类和个数
ans_dict = {}
for data in tqdm(all_data):
    ans = data['anser']
    if ans not in ans_dict.keys():
        ans_dict[ans] = 1
    else:
        ans_dict[ans] += 1

print(ans_dict)
print(len(ans_dict.keys()))
print(ans_dict.keys())

