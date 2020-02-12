# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  model_utils.py
@Description    :  
@CreateTime     :  2020/2/7 16:27
------------------------------------
@ModifyTime     :  
"""
from collections import OrderedDict
import os
import json
import logging


def config_model(FLAGS, word_to_id, tag_to_id):
    """

    :param FLAGS:
    :param word_to_id:
    :param tag_to_id:
    :return:
    """
    config = OrderedDict()
    config['num_words'] = len(word_to_id)
    config['word_dim'] = FLAGS.word_dim
    config['num_tags'] = len(tag_to_id)
    config['seg_dim'] = FLAGS.seg_dim
    config['lstm_dim'] = FLAGS.lstm_dim
    config['batch_size'] = FLAGS.batch_size
    config['optimizer'] = FLAGS.optimizer
    config['emb_file'] = FLAGS.emb_file

    config['clip'] = FLAGS.clip
    config['dropout_keep'] = 1.0 - FLAGS.dropout
    config['optimizer'] = FLAGS.optimizer
    config['lr'] = FLAGS.lr
    config['tag_scheme'] = FLAGS.tag_scheme
    config['pre_emb'] = FLAGS.pre_emb
    return config


def make_path(params):
    """
    创建文件夹
    :param params:
    :return:
    """
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)  # 结果路径
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)  # 模型保存路径
    if not os.path.isdir('log'):
        os.makedirs('log')  # 日志文件


def save_config(config, config_file):
    """
    保存配置文件
    :param config:
    :param config_path:
    :return:
    """
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    加载配置文件
    :param config_file:
    :return:
    """
    with open(config_file, encoding='utf-8') as f:
        return json.load(f)


def get_logger(log_file):
    """
    定义日志方法
    :param log_file:
    :return:
    """
    # 创建一个logging的实例 logger
    logger = logging.getLogger(log_file)
    # 设置logger的全局日志级别为DEBUG
    logger.setLevel(logging.DEBUG)
    # 创建一个日志文件的handler，并且设置日志级别为DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # 创建一个控制台的handler，并设置日志级别为DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def print_config(config, logger):
    """
    打印模型参数
    :param config:
    :param logger:
    :return:
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def main():
    pass


if __name__ == '__main__':
    main()