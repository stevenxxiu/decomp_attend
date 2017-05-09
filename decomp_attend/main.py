import csv
import datetime
import json
import os
import re
from collections import defaultdict

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.layers.core import Dense
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
                vecs.append(np.array(list(map(np.float32, words[1:]))))
                word_to_index[words[0]] = len(word_to_index)

    # visualize embeddings
    path = '__cache__/tf/emb'
    os.makedirs(path, exist_ok=True)
    config = projector.ProjectorConfig()
    emb_conf = config.embeddings.add()
    emb_conf.tensor_name = 'emb'
    emb_conf.metadata_path = os.path.abspath(os.path.join(path, 'emb_metadata.tsv'))
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
    emb = tf.Variable(tf.constant(np.array(vecs)))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver({'emb': emb})
        saver.save(sess, os.path.join(path, 'model.ckpt'), write_meta_graph=False)

    return word_to_index


def run_model(docs, word_to_index, num_unknown, embedding_size, dropout_rate, lr, batch_size, epoch_size):
    # special words
    word_to_index['\0'] = len(word_to_index)

    # network
    tf.reset_default_graph()
    X_doc_1 = tf.placeholder(tf.int32, [None, None])
    X_doc_2 = tf.placeholder(tf.int32, [None, None])
    mask_1 = tf.placeholder(tf.float32, [None, None])
    mask_2 = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool, [])
    batch_size_ = tf.shape(y)[0]

    emb = tf.Variable(tf.random_normal([len(word_to_index) + num_unknown, embedding_size], 0, 1))
    emb_1 = tf.nn.embedding_lookup(emb, X_doc_1)
    emb_2 = tf.nn.embedding_lookup(emb, X_doc_2)

    l_attend = Dense(200, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    attend_d_1 = tf.nn.relu(tf.layers.dropout(l_attend.apply(emb_1), rate=dropout_rate, training=training))
    attend_d_2 = tf.nn.relu(tf.layers.dropout(l_attend.apply(emb_2), rate=dropout_rate, training=training))
    attend_e = tf.matmul(attend_d_1, tf.transpose(attend_d_2, [0, 2, 1]))
    attend_mask_w_1 = tf.reshape(mask_1, [batch_size_, 1, -1])
    attend_mask_w_2 = tf.reshape(mask_2, [batch_size_, 1, -1])
    # attend_norm_1 are the weights to attend for in sentence 2 for each word of sentence 1
    attend_norm_1 = tf.nn.softmax(attend_mask_w_2 * attend_e + (-1 / attend_mask_w_2 + 1))
    attend_norm_2 = tf.nn.softmax(attend_mask_w_1 * tf.transpose(attend_e, [0, 2, 1]) + (-1 / attend_mask_w_1 + 1))
    attend_1 = tf.matmul(attend_norm_1, emb_2)
    attend_2 = tf.matmul(attend_norm_2, emb_1)

    l_compare = Dense(200, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    compare_1 = tf.nn.relu(tf.layers.dropout(l_compare.apply(
        tf.concat([emb_1, attend_1], 2)
    ), rate=dropout_rate, training=training))
    compare_2 = tf.nn.relu(tf.layers.dropout(l_compare.apply(
        tf.concat([emb_2, attend_2], 2)
    ), rate=dropout_rate, training=training))

    agg_1 = tf.reduce_sum(tf.reshape(mask_1, [batch_size_, -1, 1]) * compare_1, 1)
    agg_2 = tf.reduce_sum(tf.reshape(mask_2, [batch_size_, -1, 1]) * compare_2, 1)
    logits = tf.nn.relu(tf.layers.dropout(tf.layers.dense(
        tf.concat([agg_1, agg_2], 1), 200, kernel_initializer=init_ops.RandomNormal(0, 0.01)
    ), rate=dropout_rate, training=training))
    logits = tf.layers.dense(logits, 3, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # we do not need LazyAdamOptimizer since embeddings are not updated
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(loss)
    train_op = opt.apply_gradients([(grad, var) for grad, var in grads if var != emb])

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load pretrained word embeddings
        emb_0 = tf.Variable(0., validate_shape=False)
        saver = tf.train.Saver({'emb': emb_0})
        saver.restore(sess, '__cache__/tf/emb/model.ckpt')
        sess.run(emb[:tf.shape(emb_0)[0]].assign(emb_0))

        # sort so sentences have similar lengths according to paper
        parts = [[], [], []]
        for i, (sent_1, sent_2, label) in enumerate(docs):
            if len(sent_1) < 20 and len(sent_2) < 20:
                parts[0].append(i)
            elif len(sent_1) < 50 and len(sent_2) < 50:
                parts[1].append(i)
            else:
                parts[2].append(i)

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            p = np.concatenate([np.random.permutation(part) for part in parts])
            total_loss = 0
            for j in range(0, len(p), batch_size):
                k = p[j:j + batch_size]
                max_len_1 = max(len(docs[k_][0]) for k_ in k)
                max_len_2 = max(len(docs[k_][1]) for k_ in k)
                X_doc_1_ = np.zeros([len(k), max_len_1 + 1], dtype=np.int32)
                X_doc_2_ = np.zeros([len(k), max_len_2 + 1], dtype=np.int32)
                mask_1_ = np.zeros([len(k), max_len_1 + 1], dtype=np.float32)
                mask_2_ = np.zeros([len(k), max_len_2 + 1], dtype=np.float32)
                y_ = [['contradiction', 'neutral', 'entailment'].index(docs[k_][2]) for k_ in k]
                for i_k, k_ in enumerate(k):
                    doc_1 = ['\0'] + docs[k_][0]
                    doc_2 = ['\0'] + docs[k_][1]
                    X_doc_1_[i_k, :len(doc_1)] = [
                        word_to_index.get(word, len(word_to_index) + hash(word) % num_unknown) for word in doc_1
                    ]
                    X_doc_2_[i_k, :len(doc_2)] = [
                        word_to_index.get(word, len(word_to_index) + hash(word) % num_unknown) for word in doc_2
                    ]
                    mask_1_[i_k, :len(doc_1)] = 1
                    mask_2_[i_k, :len(doc_2)] = 1
                _, batch_loss = sess.run([train_op, loss], feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, y: y_, training: True,
                })
                total_loss += batch_loss
            print(datetime.datetime.now(), f'finished epoch {i}, loss: {total_loss / len(y_):f}')


def main():
    train, val, test = load_data()
    word_to_index = gen_tables(train)
    run_model(
        train, word_to_index=word_to_index, num_unknown=100, embedding_size=300, dropout_rate=0.2, lr=0.01,
        batch_size=512, epoch_size=360
    )

if __name__ == '__main__':
    main()
