import json

# 读取json文件
with open('dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取行数为3的倍数的内容
result = []
for i in range(len(data)):
    if (i + 1) % 9 == 0:
        result.append(data[i])

# 将结果写入新的json文件
with open('new_json_file_dev.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
