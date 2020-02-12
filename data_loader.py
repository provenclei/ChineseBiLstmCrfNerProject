
# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  data_loader.py
@Description    :  
@CreateTime     :  2020/2/7 16:26
------------------------------------
@ModifyTime     :  
"""
import codecs
import data_utils


def load_sentences(path):
    """
    加载数据集，每一行至少包含一个汉字和一个标记
    句子和句子之间是以空格进行分割
    最后返回句子集合
    :param path:
    :return:
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', encoding='utf-8'):
        line = line.strip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                # 非法输入
                continue
            else:
                word = line.split()
                assert len(word) >= 2, "分割之后长度大于等于2"
                sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    更新为指定编码
    :param sentence:
    :param tag_scheme:
    :return:
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # 检查输入句子编码
        if not data_utils.check_bio(tags):
            # 将这句话中的每个词使用空格拼接，再使用回车进行拼接
            s_str = "\n".join(" ".join(w) for w in s)
            raise Exception("输入的句子应为BIO编码，请检查输入句子 %i:\n%s" % (i, s_str))

        if tag_scheme == "BIO":
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag

        if tag_scheme == "BIOES":
            new_tags = data_utils.bio_to_bioes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception("非法目标编码")


def word_mapping(sentences):
    """
    构建字典
    :param sentences:
    :return:
    """
    word_list = [[item[0] for item in sentence] for sentence in sentences]
    word_dict = data_utils.create_dict(word_list)
    word_dict['<PAD>'] = 100001
    word_dict['<UNK>'] = 100000
    word_to_id, id_to_word = data_utils.create_mapping(word_dict)
    return word_dict, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    构建标签字典
    :param sentences:
    :return:
    """
    tag_list = [[item[1] for item in sentence] for sentence in sentences]
    tag_dict = data_utils.create_dict(tag_list)
    tag_to_id, id_to_tag = data_utils.create_mapping(tag_dict)
    return tag_dict, tag_to_id, id_to_tag


def prepare_dataset(sentences, word_to_id, tag_to_id, train=True):
    """
    数据预处理，返回list其实包含
    -word_list
    -word_id_list
    -word char indexs
    -tag_id_list
    :param sentences:
    :param word_to_id:
    :param tag_to_id:
    :param train:
    :return:
    """
    non_index = tag_to_id['O']
    data = []

    for s in sentences:
        word_list = [w[0] for w in s]
        word_id_list = [word_to_id[w if w in word_to_id else '<UNK>'] for w in word_list]
        segs = data_utils.get_seg_features("".join(word_list))
        if train:
            tag_id_list = [tag_to_id[w[-1]] for w in s]
        else:
            tag_id_list = [non_index for w in s]
        data.append([word_list, word_id_list, segs, tag_id_list])
    return data


def main():
    path = 'data/ner.dev'
    sentences = load_sentences(path)
    update_tag_scheme(sentences, "BIOES")
    _, word_to_id, id_to_word = word_mapping(sentences)
    _, tag_to_id, id_to_tag = tag_mapping(sentences)
    dev_data = prepare_dataset(sentences, word_to_id, tag_to_id)
    data_utils.BatchManager(dev_data, 128)
    print("load sentence over")


if __name__ == '__main__':
    main()