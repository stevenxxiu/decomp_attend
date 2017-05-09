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
from tensorflow.python.layers.core import Dense, Dropout

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
    # network
    tf.reset_default_graph()
    X_doc_1 = tf.placeholder(tf.int32, [None, None])
    X_doc_2 = tf.placeholder(tf.int32, [None, None])
    mask_1 = tf.placeholder(tf.float32, [None, None])
    mask_2 = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool, [])

    emb = tf.Variable(tf.random_normal([len(word_to_index) + num_unknown, embedding_size], 0, 1))
    emb_1 = tf.nn.embedding_lookup(emb, X_doc_1)
    emb_2 = tf.nn.embedding_lookup(emb, X_doc_2)

    l_attend = Dense(200, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    attend_d_1 = tf.nn.relu(tf.layers.dropout(l_attend.apply(emb_1), rate=dropout_rate, training=training))
    attend_d_2 = tf.nn.relu(tf.layers.dropout(l_attend.apply(emb_2), rate=dropout_rate, training=training))
    attend_e = tf.matmul(attend_d_1, tf.transpose(attend_d_2, [0, 2, 1]))
    attend_mask_w_1 = tf.reshape(mask_1, [batch_size, 1, -1])
    attend_mask_w_2 = tf.reshape(mask_2, [batch_size, 1, -1])
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

    agg_1 = tf.reduce_sum(tf.reshape(mask_1, [batch_size, -1, 1]) * compare_1, 1)
    agg_2 = tf.reduce_sum(tf.reshape(mask_2, [batch_size, -1, 1]) * compare_2, 1)
    logits = tf.nn.relu(tf.layers.dropout(tf.layers.dense(
        tf.concat([agg_1, agg_2], 1), 200, kernel_initializer=init_ops.RandomNormal(0, 0.01)
    ), rate=dropout_rate, training=training))
    logits = tf.layers.dense(logits, 3, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_op = LazyAdamOptimizer(learning_rate=lr).minimize(loss)

    # XXX do not train embeddings

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pass

        # XXX prepend each sentence with null token
        # XXX sort so sentences have similar lengths according to paper

        # # train
        # print(datetime.datetime.now(), 'started training')
        # q = Queue(1)
        # Process(target=dbow_sample, args=(docs, word_to_index, word_to_freq, sample, epoch_size, q)).start()
        # for i in range(epoch_size):
        #     X_doc_, y_ = q.get()
        #     p = np.random.permutation(len(y_))
        #     total_loss = 0
        #     for j in range(0, len(y_), batch_size):
        #         k = p[j:j + batch_size]
        #         _, batch_loss = sess.run([train_op, loss], feed_dict={X_doc: X_doc_[k], y: y_[k]})
        #         total_loss += batch_loss
        #     print(datetime.datetime.now(), f'finished epoch {i}, loss: {total_loss / len(y_):f}')
        #
        # # save
        # path = os.path.join('__cache__', 'tf', f'dbow-{name}-{uuid.uuid4()}')
        # os.makedirs(path)
        # save_model(path, docs, word_to_index, word_to_freq, emb_doc, None, l.W, sess)
        # return path


def main():
    train, val, test = load_data()
    word_to_index = gen_tables(train)
    run_model(
        train, word_to_index=word_to_index, num_unknown=100, embedding_size=300, dropout_rate=0.2, lr=0.01,
        batch_size=2, epoch_size=360
    )


if __name__ == '__main__':
    main()
