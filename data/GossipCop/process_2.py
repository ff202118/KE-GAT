import json
import wikipediaapi
import time

user_agent = 'your-email@example.com'
wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent)

# # 去除重复的实体词
# entity = set()
# entity_num = 0
# count = 0
# datasets = ['train', 'test', 'val']

# while count < 3:
#     dataset = datasets[count]
#     with open(f'process/{dataset}.json', 'r') as f:
#         List = json.load(f)
#
#         for Dict in List:
#             for name in Dict['entity']:
#                 entity.add(name)
#                 entity_num += 1
#     count += 1
#
# print(len(entity), entity_num)
#
# entity = list(entity)
# entity_title = '\n'.join(entity)
# with open('process/desc/entity_title.txt', 'w', encoding='utf-8') as f:
#         f.write(entity_title)

# 去重后，获取实体描述
max_retries = 3  # 设置最大重试次数
retry_delay = 5  # 设置重试间隔为 5 秒
write_interval = 10  # 设置写入间隔为每完成 30 个字典
List_desc = []
entity = []
with open('process/desc/entity_title.txt', 'r', encoding='utf-8') as f:
    for line in f:
        entity.append(line.strip())


entity_len = len(entity)
desc_len = 0

for i, name in enumerate(entity, 1):
    Dict_desc = {}
    if name == "":
        continue
    retry_count = 0
    start_time = time.time()  # 记录开始时间
    while retry_count < max_retries:
        try:
            # 尝试获取维基百科页面摘要
            page_py = wiki_wiki.page(name)
            summary = page_py.summary
            Dict_desc[f"{name}"] = summary
            desc_len += 1
            break  # 如果成功获取摘要，则跳出循环
        except Exception as e:
            # 如果出现异常，记录异常信息并增加重试计数
            print(f"Failed to fetch summary for entity '{name}' on attempt {retry_count + 1}: {e}")
            retry_count += 1
            time.sleep(retry_delay)  # 等待重试间隔
        if time.time() - start_time > 10:  # 检查是否超时
            print(f"Timeout occurred for entity '{name}'")
            continue  # 超时则跳过当前实体的处理
    else:
        # 如果重试次数达到最大值仍然无法获取摘要，添加空字符串到列表中
        print(f"Failed to fetch summary for entity '{name}' after {max_retries} attempts")
        Dict_desc[f"{name}"] = ""
        continue

    List_desc.append(Dict_desc)
    # print(List_desc)
    # 每完成30个字典的处理就写入一次
    if i % write_interval == 0:
        with open(f'process/desc/entity_descpartial.json', 'w') as f:
            json.dump(List_desc[:i], f)

# 最后将整个列表写入到文件中
with open('process/desc/entity_desc.json', 'w') as f:
    json.dump(List_desc, f)

print(f"entity_len: {entity_len}, desc_len: {desc_len}")
