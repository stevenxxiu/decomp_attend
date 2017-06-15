import argparse
import csv
import datetime
import inspect
import json
import os
import re
import shlex
import sys
from collections import defaultdict
from multiprocessing import Process, Queue

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.layers.core import Dense, Dropout
from tensorflow.python.ops import init_ops

memory = joblib.Memory('__cache__', verbose=0)


def load_data():
    res = []
    for file in ('snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl'):
        with open(f'../../data/stanford_natural_language_inference/{file}', 'r', encoding='utf-8') as sr:
            cur_res = []
            for line in sr:
                obj = json.loads(line)
                if obj['gold_label'] == '-':
                    continue
                sent_1 = re.sub(r'[()]', '', obj['sentence1_binary_parse']).split()
                sent_2 = re.sub(r'[()]', '', obj['sentence2_binary_parse']).split()
                cur_res.append((sent_1, sent_2, obj['gold_label']))
            res.append(cur_res)
    return res


@memory.cache(ignore=['train', 'val', 'test'])
def gen_tables(train, val, test):
    # load required glove vectors
    word_to_freq = defaultdict(int)
    for docs in train, val, test:
        for sent1, sent2, label in docs:
            for sent in sent1, sent2:
                for word in sent:
                    word_to_freq[word] += 1
    vecs = []
    word_to_index = {}
    with open('../../data/glove/glove.840B.300d.txt', 'r', encoding='utf-8') as sr:
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


def apply_layers(layers, input_, **kwargs):
    for layer in layers:
        names = inspect.signature(layer.call).parameters
        input_ = layer.apply(input_, **{name: arg for name, arg in kwargs.items() if name in names})
    return input_


def sample(docs, word_to_index, num_unknown, epoch_size, batch_size, q):
    # sort so sentences have similar lengths according to paper
    parts = [[], [], []]
    for i, (sent_1, sent_2, label) in enumerate(docs):
        if len(sent_1) < 20 and len(sent_2) < 20:
            parts[0].append(i)
        elif len(sent_1) < 50 and len(sent_2) < 50:
            parts[1].append(i)
        else:
            parts[2].append(i)
    for i in range(epoch_size):
        res = []
        p = np.concatenate([np.random.permutation(part) for part in parts])
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
            res.append((X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_))
        q.put(res)


def attend_intra(w, emb, mask, n_intra_bias, long_dist_bias, dist_biases):
    i = tf.range(0, tf.shape(mask)[1], dtype=tf.int32)
    ij = tf.expand_dims(i, 1) - tf.expand_dims(i, 0)
    ij_mask = tf.cast(tf.logical_and(tf.less_equal(ij, n_intra_bias), tf.greater_equal(ij, -n_intra_bias)), tf.float32)
    w = w + (1 - ij_mask) * long_dist_bias + ij_mask * tf.gather(dist_biases, ij + n_intra_bias)
    mask = tf.expand_dims(mask, 1)
    norm = tf.nn.softmax(mask * w + (-1 / mask + 1))
    return tf.matmul(norm, emb)


def attend_inter(w, emb, mask):
    mask = tf.expand_dims(mask, 1)
    norm = tf.nn.softmax(mask * w + (-1 / mask + 1))
    return tf.matmul(norm, emb)


# noinspection PyTypeChecker
def run_model(
    train, val, test, word_to_index, intra_sent, emb_unknown, emb_size, emb_normalize, emb_proj,
    emb_proj_pca, n_intra, n_intra_bias, n_attend, n_compare, n_classif, dropout_rate, lr, batch_size, epoch_size
):
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

    emb = tf.Variable(tf.random_normal([len(word_to_index) + emb_unknown, emb_size], 0, 1))
    l_proj_emb = Dense(emb_proj, use_bias=False, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    emb_ = [None, None]
    for i in range(2):
        emb_[i] = tf.nn.embedding_lookup(emb, [X_doc_1, X_doc_2][i])
        if emb_proj:
            emb_[i] = l_proj_emb.apply(emb_[i])

    if intra_sent:
        l_intra = sum([[
            Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
            Dropout(rate=dropout_rate),
        ] for n in n_intra], [])
        long_dist_bias = tf.Variable(tf.zeros([]))
        dist_biases = tf.Variable(tf.zeros([2 * n_intra_bias + 1]))
        for i in range(2):
            intra_d = apply_layers(l_intra, emb_[i], training=training)
            intra_w = tf.matmul(intra_d, tf.transpose(intra_d, [0, 2, 1]))
            emb_[i] = tf.concat([emb_[i], attend_intra(
                intra_w, emb_[i], [mask_1, mask_2][i], n_intra_bias, long_dist_bias, dist_biases
            )], 2)

    l_attend = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_attend], [])
    attend_d_1 = apply_layers(l_attend, emb_[0], training=training)
    attend_d_2 = apply_layers(l_attend, emb_[1], training=training)
    attend_w = tf.matmul(attend_d_1, tf.transpose(attend_d_2, [0, 2, 1]))
    attend_1 = attend_inter(attend_w, emb_[1], mask_2)
    attend_2 = attend_inter(tf.transpose(attend_w, [0, 2, 1]), emb_[0], mask_1)

    l_compare = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_compare], [])
    compare_1 = apply_layers(l_compare, tf.concat([emb_[0], attend_1], 2), training=training)
    compare_2 = apply_layers(l_compare, tf.concat([emb_[1], attend_2], 2), training=training)

    agg_1 = tf.reduce_sum(tf.expand_dims(mask_1, -1) * compare_1, 1)
    agg_2 = tf.reduce_sum(tf.expand_dims(mask_2, -1) * compare_2, 1)
    l_classif = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_classif], [])
    logits = apply_layers(l_classif, tf.concat([agg_1, agg_2], 1), training=training)
    logits = tf.layers.dense(logits, 3, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # we do not need LazyAdamOptimizer since embeddings are not updated
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(loss)
    train_op = opt.apply_gradients([(grad, var) for grad, var in grads if var != emb])

    # run
    with tf.Session() as sess:
        # start sampling
        qs = {name: Queue(1) for name in ('train', 'val', 'test')}
        for name, docs in ('train', train), ('val', val), ('test', test):
            Process(target=sample, args=(docs, word_to_index, emb_unknown, epoch_size, batch_size, qs[name])).start()

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # load pretrained word embeddings
        emb_0 = tf.Variable(0., validate_shape=False)
        saver = tf.train.Saver({'emb': emb_0})
        saver.restore(sess, '__cache__/tf/emb/model.ckpt')

        # embedding transforms
        sess.run(emb[:tf.shape(emb_0)[0]].assign(emb_0))
        if emb_normalize:
            sess.run(emb.assign(emb / tf.reshape(tf.norm(emb, axis=1), [-1, 1])))
        if emb_proj_pca:
            # centering has no effect as bias can be absorbed in the next layer's bias
            sess.run(emb.assign(emb - tf.reshape(tf.reduce_mean(emb, axis=0), [1, -1])))
            sess.run(l_proj_emb.kernel.assign(tf.svd(emb)[2][:, :emb_proj]))

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            total_loss, correct = 0, {'val': 0, 'test': 0}
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in qs['train'].get():
                _, batch_loss = sess.run([train_op, loss], feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, y: y_, training: True,
                })
                total_loss += len(y_) * batch_loss
            for name in 'val', 'test':
                for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in qs[name].get():
                    logits_ = sess.run(logits, feed_dict={
                        X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, training: False,
                    })
                    correct[name] += np.sum(np.argmax(logits_, 1) == y_)
            print(
                datetime.datetime.now(),
                f'finished epoch {i}, loss: {total_loss / len(train):f}, '
                f'val acc: {correct["val"] / len(val):f}, test acc: {correct["test"] / len(test):f}'
            )


def main():
    print(' '.join(shlex.quote(arg) for arg in sys.argv[1:]))
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('hyperparams')
    args = arg_parser.parse_args()
    train, val, test = load_data()
    word_to_index = gen_tables(train, val, test)
    run_model(train, val, test, word_to_index=word_to_index, **json.loads(args.hyperparams))

if __name__ == '__main__':
    main()
