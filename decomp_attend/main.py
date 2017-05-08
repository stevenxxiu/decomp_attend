import csv
import os
import json
import re
from collections import defaultdict

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.contrib.opt import LazyAdamOptimizer
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import init_ops

memory = joblib.Memory('__cache__', verbose=0)


def load_data():
    res = []
    for file in ('snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl'):
        with open(f'../data/stanford_natural_language_inference/{file}', 'r', encoding='utf-8') as sr:
            cur_res = []
            for line in sr:
                obj = json.loads(line)
                if obj['gold_label'] == '-':
                    continue
                sent1 = re.sub(r'[()]', '', obj['sentence1_binary_parse']).split()
                sent2 = re.sub(r'[()]', '', obj['sentence2_binary_parse']).split()
                cur_res.append((sent1, sent2, obj['gold_label']))
            res.append(cur_res)
    return res


@memory.cache(ignore=['docs'])
def gen_tables(docs):
    # load required glove vectors
    word_to_freq = defaultdict(int)
    for sent1, sent2, label in docs:
        for sent in sent1, sent2:
            for word in sent:
                word_to_freq[word] += 1
    vecs = []
    word_to_index = {}
    with open('../data/glove/glove.840B.300d.txt', 'r', encoding='utf-8') as sr:
        for line in sr:
            words = line.split(' ')
            if words[0] in word_to_freq:
                vecs.append(np.array(list(map(float, words[1:]))))
                word_to_index[words[0]] = len(word_to_index)

    # visualize embeddings
    path = '__cache__/tf/emb_word'
    os.makedirs(path, exist_ok=True)
    config = projector.ProjectorConfig()
    emb_conf = config.embeddings.add()
    emb_conf.tensor_name = 'emb_word'
    emb_conf.metadata_path = os.path.abspath(os.path.join(path, 'emb_word_metadata.tsv'))
    with open(emb_conf.metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['Word', 'Frequency'])
        words = len(word_to_index) * [None]
        for word, i in word_to_index.items():
            words[i] = word
        for word in words:
            writer.writerow([word, word_to_freq[word]])
    summary_writer = tf.summary.FileWriter(path)
    projector.visualize_embeddings(summary_writer, config)
    emb_word = tf.Variable(tf.constant(np.array(vecs)))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver({'emb_word': emb_word})
        saver.save(sess, os.path.join(path, 'model.ckpt'))

    return word_to_index


def run_model():
    pass


def main():
    train, val, test = load_data()
    gen_tables(train)


if __name__ == '__main__':
    main()
