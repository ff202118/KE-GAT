import json
import re
# import spacy
from collections import defaultdict, Counter
import numpy as np


class StringProcess(object):
    """
        Tokenization/string cleaning for the SST yelp_dataset
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        # self.url = re.compile(r"[a-z]*[:.]+\S+|\n|\s+", flags=0)
        self.url = re.compile(
            r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        string = re.sub(r"\[\d+\]", "", string)  # 匹配形如[1]、[2]之类的模式，并将其替换为空字符串
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r",+", ", ", string)

        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result

# 去除 低频词
def remove_less_word(lines_str, word_st):
    return " ".join([word for word in lines_str.split() if word in word_st])

# 小写化，去除 低频词 和 停用词
def clean_data(dataset):
    with open(f'data/GossipCop/process/{dataset}.json') as f:
        List = json.load(f)

    sp = StringProcess()
    word_list = []
    # 统计所有词（去重）
    for Dict in List:
        text = Dict['text'].strip()
        clean_text = sp.remove_stopword(sp.clean_str(text))

        word_list.extend(clean_text.split())

    # 统计高频词
    word_set = set()
    for word, value in Counter(word_list).items():
        if value < 5:
            continue
        word_set.add(word)

    # 清理文本
    def clean(data):
        data = sp.clean_str(data.strip())
        data = sp.remove_stopword(data)
        data = remove_less_word(data, word_set)

        return data

    # 统计文本长度
    text_len_list = []
    summary_len_list = []

    for Dict in List:
        Dict['text'] = clean(Dict['text'])
        Dict['summary'] = clean(Dict['summary'])

        text_len_list.append(len(Dict['text'].split()))
        summary_len_list.append(len(Dict['summary'].split()))

    with open(f'data/GossipCop/clean/{dataset}.json', 'w') as f:
        json.dump(List, f)

    print(f"text average length:{np.mean(text_len_list)}, summary average length:{np.mean(summary_len_list)}")
    print(f"text count:{len(text_len_list)}, summary count:{len(summary_len_list)}")
    print("Total number of words:", len(word_set))

if __name__ == "__main__":
    dataset = "test"
    clean_data(dataset)




