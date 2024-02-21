import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

with open('data/GossipCop/clean/test.json', 'r') as f:
    List = json.load(f)

real = []
fake = []

# for Dict in List:
#     news_content = Dict['text']
#     summary = Dict['summary']
#     label = Dict['label'].lower()
#
#     # 使用TF-IDF向量化器对新闻内容和摘要进行向量化
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([news_content, summary])
#
#     # 计算余弦相似度
#     cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
#
#     if label == "real":
#         real.append(cosine_sim[0][0])
#     else:
#         fake.append(cosine_sim[0][0])

for Dict in List:
    emotion = Dict['emotion']
    label = Dict['label'].lower()

    if label == "real":
        real.append(emotion)
    else:
        fake.append(emotion)

real = np.array(real)
fake = np.array(fake)

real_mean = np.mean(real, axis=0)
fake_mean = np.mean(fake, axis=0)
# print(f"real：{np.mean(real)}, {len(real)}, {max(real)}; fake：{np.mean(fake)}, {len(fake)}, {max(fake)}")
# print(real)
# print(fake)
print(f"real：{real_mean}; fake：{fake_mean}")
