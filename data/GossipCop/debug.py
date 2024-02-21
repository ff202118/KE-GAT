"""
    从新闻文本中提取实体，摘要，情感，标题
"""

from nltk.tokenize import sent_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import tagme
from nltk.corpus import stopwords
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json

# 提供 TagMe API 密钥
tagme.GCUBE_TOKEN = "813a5b09-e661-4799-9344-67bec90a72f9-843339462"

"""
    使用TextRank提取 文本摘要
"""
def sentence_similarity(sent1, sent2, stopwords=None):
    try:
        if stopwords is None:
            stopwords = []

        sent1 = [word.lower() for word in sent1.split() if word.lower() not in stopwords]
        sent2 = [word.lower() for word in sent2.split() if word.lower() not in stopwords]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        for word in sent1:
            vector1[all_words.index(word)] += 1

        for word in sent2:
            vector2[all_words.index(word)] += 1

        return 1 - cosine_distance(vector1, vector2)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def build_similarity_matrix(sentences, stop_words):
    try:
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def generate_summary(text, top_n=1):
    try:
        stop_words = set(stopwords.words('english'))
        summarize_text = []

        sentences = sent_tokenize(text)

        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        for i in range(top_n):
            summarize_text.append(ranked_sentence[i][1])

        return ". ".join(summarize_text)
    except:
        return ""

"""
    使用 tagme 和 wiki 获取 实体 和 实体描述
"""
# def get_entity(text):
#     # 创建 Wikipedia 对象，仅传递 user_agent 参数一次
#     # wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent)
#     try:
#         annotations = tagme.annotate(text)
#
#         # entity_desc_list = []
#         entity = set()
#
#         # 提取命名实体并进行实体链接
#         for annotation in annotations.get_annotations(0.4):
#             entity_title = annotation.entity_title
#             # entity_uri = tagme.title_to_uri(entity_title)
#             # print("Entity:", entity_title, " - Score:", annotation.score, " - Wikipedia URL:", entity_uri)
#             entity.add(entity_title)
#
#         # for name in entity:
#         #     max_retries = 3
#         #     retry_count = 0
#         #     while retry_count < max_retries:
#         #         try:
#         #             # 尝试获取维基百科页面摘要
#         #             page_py = wiki_wiki.page(name)
#         #             summary = page_py.summary
#         #             entity_desc_list.append(summary)
#         #             break  # 如果成功获取摘要，则跳出循环
#         #         except Exception as e:
#         #             # 如果出现异常，记录异常信息并增加重试计数
#         #             print(f"Failed to fetch summary for entity '{name}' on attempt {retry_count + 1}: {e}")
#         #             retry_count += 1
#         #     else:
#         #         # 如果重试次数达到最大值仍然无法获取摘要，添加空字符串到列表中
#         #         print(f"Failed to fetch summary for entity '{name}' after {max_retries} attempts")
#         #         entity_desc_list.append("")
#
#         return list(entity)
#     except:
#         return []

from concurrent.futures import ThreadPoolExecutor, TimeoutError

def get_entity(text):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(tagme.annotate, text)
        try:
            annotations = future.result(timeout=10)  # 设置超时时间为10秒
            entity = set()

            for annotation in annotations.get_annotations(0.4):
                entity_title = annotation.entity_title
                entity.add(entity_title)

            return list(entity)
        except TimeoutError:
            print("Function timed out")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []


"""
    获取新闻情感
"""
def get_emotion(text, semantic_cls):
    try:
        result = semantic_cls(text)
        return result['scores']
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def main(text):
    try:
        semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_bert_sentiment-analysis_english-base')

        summary = generate_summary(text)
        entity = get_entity(text)
        emotion = get_emotion(text, semantic_cls)
        # print(entity)
        return summary, entity, emotion
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", [], []


# # 测试集
# id = 0  # 初始化 id 变量
#
# with open('test.json', 'r') as f:
#     List = json.load(f)
#
#     for i, Dict in enumerate(List, 1):
#         ok = 'success'
#         id += 1  # 每次迭代递增 id
#         summary, entity, emotion = main(Dict['text'])
#         Dict['summary'] = summary
#         Dict['entity'] = entity
#         # Dict['entity_desc'] = entity_desc
#         Dict['emotion'] = emotion
#         Dict['id'] = id
#         if entity == []:
#             ok = 'fail'
#         print(id, ok)
#
#         # 每完成30个字典的处理就写入一次
#         if i % 30 == 0:
#             with open('process/test_partial.json', 'w') as f:
#                 json.dump(List[:i], f)
#
# # 最后将整个列表写入到文件中
# with open('process/test.json', 'w') as f:
#     json.dump(List, f)

# # 验证集
id = 3620  # 初始化 id 变量

with open('process/desc/train.json', 'r') as f:
    List = json.load(f)

    for i, Dict in enumerate(List, 1):
        ok = 'success'
        id += 1  # 每次迭代递增 id
        summary, entity, emotion = main(Dict['text'])
        Dict['summary'] = summary
        Dict['entity'] = entity
        # Dict['entity_desc'] = entity_desc
        Dict['emotion'] = emotion
        Dict['id'] = id
        if entity == []:
            ok = 'fail'
        print(id, ok)

        # 每完成30个字典的处理就写入一次
        if i % 30 == 0:
            with open('process/desc/train_partial2.json', 'w') as f:
                json.dump(List[:i], f)

# 最后将整个列表写入到文件中
with open('process/train.json', 'w') as f:
    json.dump(List, f)