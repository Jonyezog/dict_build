# -*- coding: utf-8 -*-

import re

def entropy_of_list(ls):
    """
    给定列表，计算熵值
    The entropy is sum of -p[i] * log(p[i]) for every unique element i in the list, and p[i] is its frequency
    """
    elements = {}
    for e in ls:
        elements[e] = elements.get(e, 0) + 1
    length = float(len(ls))
    return sum(map(lambda v: -v / length * math.log(v / length), elements.values()))

def index_of_sorted_suffix(doc, max_word_len):
    indexes = []
    length = len(doc)
    for i in range(0, length):
        for j in range(i + 1, min(i + 1 + max_word_len, length + 1)):
            indexes.append((i, j))
    # 根据内容排序
    return sorted(indexes, key=lambda m: doc[m[0]:m[1]])


def gen_sub_parts(string):
    """
    将字符串分成可能的两部分
    给定 "abcd", 生成[("a", "bcd"), ("ab", "cd"), ("abc", "d")]
    """
    length = len(string)
    res = []
    for i in range(1, length):
        res.append((string[0:i], string[i:]))
    return res

class WordInfo(object):
    """
    保存每个词的信息，包括频率、左邻居和右邻居词
    """
    def __init__(self, text):
        super(WordInfo, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = []
        self.right = []
        self.cohesion = 0
        self.pp = 0.0

    def update(self, left, right):
        self.freq += 1
        if left:
            self.left.append(left)
        if right:
            self.right.append(right)

    def compute(self, length):
        """
        计算词频概率和左右信息熵
        """
        self.freq /= length
        self.left = entropy_of_list(self.left)
        self.right = entropy_of_list(self.right)

    def compute_cohesion(self, words_dict):
        """
        计算词的内凝性
        """
        parts = gen_sub_parts(self.text)
        if len(parts) > 0:
            self.cohesion = min(map(
                lambda p: self.freq / words_dict[p[0]].freq / words_dict[p[1]].freq,
                parts))

    def compute_pp(self, pos_prop):
        prefix = self.text[0]
        suffix = self.text[-1]
        if pos_prop.get(prefix) is not None and pos_prop.get(suffix) is not None:
            self.pp = min(pos_prop.get(prefix)[0], pos_prop.get(suffix)[2])


class NewWordFinder(object):
    def __init__(self, doc, pos_prop_path, max_word_len=5, min_freq=0.00005, min_entropy=2.0, min_cohesion=10.0, pp=0.1):
        super(NewWordFinder, self).__init__()
        self.max_word_len = max_word_len
        self.min_freq = min_freq
        self.pp = pp
        self.min_entropy = min_entropy
        self.min_cohesion = min_cohesion
        self.pos_prop = self.read_pos_prop(pos_prop_path)
        self.word_info = self.gen_words(doc)

        # 查找新词的过滤条件
        def filter_func(v):
            return len(v.text) > 1 and v.cohesion > self.min_cohesion and \
                    v.freq > self.min_freq and v.left > self.min_entropy and \
                    v.right > self.min_entropy and v.pp > self.pp

        self.new_words = map(lambda w: (w.text, w.freq, w.left, w.right, w.cohesion, w.pp),
                             filter(filter_func, self.word_info))

    @staticmethod
    def read_pos_prop(path):
        result = dict()
        with open(path, 'r') as pos_prop_file:
            for line in pos_prop_file.readlines():
                line = line.strip()
                eles = line.split("\t")
                pos_p = list()
                pos_p.extend(float(ele) for ele in eles[1:])
                result[eles[0]] = pos_p
        return result

    def gen_words(self, doc):
        """
        Generate all candidate words with their frequency/entropy/cohesion/pos_prop information
        """
        pattern = re.compile(u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
        doc = re.sub(pattern, ' ', doc)
        suffix_indexes = index_of_sorted_suffix(doc, self.max_word_len)
        word_cands = {}
        # compute frequency and neighbors
        for suf in suffix_indexes:
            word = doc[suf[0]:suf[1]]
            if word not in word_cands:
                word_cands[word] = WordInfo(word)
            word_cands[word].update(doc[suf[0] - 1:suf[0]], doc[suf[1]:suf[1] + 1])
        # compute probability and entropy
        length = len(doc)
        for k in word_cands:
            word_cands[k].compute(length)
            word_cands[k].compute_pp(self.pos_prop)
        # compute aggregation of words whose length > 1
        values = sorted(word_cands.values(), key=lambda x: len(x.text))
        for v in values:
            if len(v.text) == 1:
                continue
            v.compute_cohesion(word_cands)

        return sorted(values, key=lambda v: v.freq, reverse=True)

if __name__ == '__main__':
    doc = '十四是十四四十是四十，，十四不是四十，，，，四十不是十四'
    cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    pos_prop_path = os.path.join(cur_dir, 'datas', 'common', 'pos_prop.txt')
    ws = NewWordFinder(doc, pos_prop_path, max_word_len=4,
        min_freq=0.001, min_cohesion=1.0, min_entropy=0.5, pp=0.01)
    print('\n'.join(map(lambda w: '《%s》 freq:%f,left_entropy:%f,
        right_entropy:%f,cohesion:%f,pos_prop:%f' % w, ws.new_words)))


