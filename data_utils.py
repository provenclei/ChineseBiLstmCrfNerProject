# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  data_utils.py
@Description    :  
@CreateTime     :  2020/2/7 16:27
------------------------------------
@ModifyTime     :  
"""
import jieba
import math
import random


def bio_to_bioes(tags):
    """
    把bio编码转换成bioes编码
    返回新的tags
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            # 当前tag不是最后一个并且后边有I
            if (i+1) < len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B', 'S'))
        elif tag.split('-')[0] == 'I':
            if (i+1) < len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I', 'E'))
        else:
            raise Exception("非法编码！！！tags:%s, i:%i, tag:%s" % (tags, i, tag))
    return new_tags


def bioes_to_bio(tags):
    """
    BIOES->BIO
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == "B":
            new_tags.append(tag)
        elif tag.split('-')[0] == "I":
            new_tags.append(tag)
        elif tag.split('-')[0] == "S":
            new_tags.append(tag.replace('S-','B-'))
        elif tag.split('-')[0] == "E":
            new_tags.append(tag.replace('E-','I-'))
        elif tag.split('-')[0] == "O":
            new_tags.append(tag)
        else:
            raise Exception('非法编码格式')
    return new_tags


def check_bio(tags):
    """
    检测输入的tags是否是bio编码
    如果不是bio编码
    那么错误的类型
    (1)编码不在BIO中
    (2)第一个编码是I
    (3)当前编码不是B,前一个编码不是O
    :param tags:
    :return:
    """
    for i, tag in enumerate(tags):
        # tags:['O', 'O', 'B-ORG', 'I-ORG']
        # tag_list:['B', 'ORG']
        # tag:'B-ORG'
        if tag == "O":
            continue
        tag_list = tag.split("-")
        if len(tag_list) != 2 or tag_list[0] not in set(['B', 'I']):
            return False
        if tag_list[0] == 'B':
            continue
        elif i == 0 or tags[i-1] == 'O':
            # 如果第一个位置不是B或者当前编码不是B并且前一个编码0，则全部转换成B
            tags[i] = 'B' + tag[1:]
        elif tags[i-1][1:] == tag[1:]:
            # 如果当前编码的后面类型编码与tags中的前一个编码中后面类型编码相同则跳过,例如：'-ORG'
            continue
        else:
            # 如果编码类型不一致，则重新从B开始编码
            tags[i] = 'B' + tag[1:]
    return True


def create_dict(word_list):
    """
    对于word_list中的每一个items，统计items中item在word_list中的次数
    item:出现的次数
    :param word_list:
    :return:
    """
    assert type(word_list) is list
    dict = {}
    for items in word_list:
        for item in items:
            if item not in dict:
                dict[item] = 1
            else:
                dict[item] += 1
    return dict


def create_mapping(word_dict):
    """
    创建item_to_id, id_to_item
    item的排序按词典中出现的次数
    :param word_dict:
    :return:
    """
    sort_items = sorted(word_dict.items(), key=lambda x: (x[1], x[0]), reverse=True) # 返回一个数组
    id_to_word = {i: v[0] for i, v in enumerate(sort_items)}
    word_to_id = {v: k for k, v in id_to_word.items()}
    return word_to_id, id_to_word


def get_seg_features(words):
    """
    利用jieba分词
    采用类似bioes的编码，0表示单个字成词, 1表示一个词的开始， 2表示一个词的中间，3表示一个词的结尾
    :param words:
    :return:
    """
    seg_features = []
    word_list = list(jieba.cut(words))
    for word in word_list:
        if len(word) == 1:
            seg_features.append(0)
        else:
            temp = [2] * len(word)
            temp[0] = -1
            temp[-1] = 3
            seg_features.extend(temp)
    return seg_features


class BatchManager(object):
    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data)/batch_size))
        sorted_data = sorted(data, key=lambda x:len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size:(i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        word_list = []
        word_id_list = []
        seg_list = []
        tag_id_list = []
        max_length = max(len(sen[0]) for sen in data)
        for line in data:
            words, word_ids, segs, tag_ids = line
            padding = [0]*(max_length - len(words))
            word_list.append(words + padding)
            word_id_list.append(word_ids + padding)
            seg_list.append(segs + padding)
            tag_id_list.append(tag_ids + padding)
        return [word_list, word_id_list, seg_list,tag_id_list]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

