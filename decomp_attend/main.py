import csv
import datetime
import inspect
import json
import os
import re
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
        with open(f'../data/stanford_natural_language_inference/{file}', 'r', encoding='utf-8') as sr:
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


def run_model(
    train, val, test, word_to_index, intra_sent, emb_unknown, emb_size, emb_center, emb_normalize, emb_proj,
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
    batch_size_ = tf.shape(X_doc_1)[0]

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
        dist_biases = tf.Variable(tf.zeros([2 * n_intra_bias + 1]))
        for i in range(2):
            intra_d = apply_layers(l_intra, emb_[i], training=training)
            intra_i = tf.range(0, tf.shape([X_doc_1, X_doc_2][i])[1], dtype=tf.int32)
            intra_ij = tf.reshape(intra_i, [-1, 1]) - tf.reshape(intra_i, [1, -1])
            intra_ij = tf.clip_by_value(intra_ij + n_intra_bias, 0, 2 * n_intra_bias)
            intra_w = tf.matmul(intra_d, tf.transpose(intra_d, [0, 2, 1])) + tf.gather(dist_biases, intra_ij)
            intra_mask = tf.reshape([mask_1, mask_2][i], [batch_size_, 1, -1])
            intra_norm = tf.nn.softmax(intra_mask * intra_w + (-1 / intra_mask + 1))
            intra = tf.matmul(intra_norm, emb_[i])
            emb_[i] = tf.concat([emb_[i], intra], 2)

    l_attend = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_attend], [])
    attend_d_1 = apply_layers(l_attend, emb_[0], training=training)
    attend_d_2 = apply_layers(l_attend, emb_[1], training=training)
    attend_w = tf.matmul(attend_d_1, tf.transpose(attend_d_2, [0, 2, 1]))
    attend_mask_w_1 = tf.reshape(mask_1, [batch_size_, 1, -1])
    attend_mask_w_2 = tf.reshape(mask_2, [batch_size_, 1, -1])
    # attend_norm_1 are the weights to attend for in sentence 2 for each word of sentence 1
    attend_norm_1 = tf.nn.softmax(attend_mask_w_2 * attend_w + (-1 / attend_mask_w_2 + 1))
    attend_norm_2 = tf.nn.softmax(attend_mask_w_1 * tf.transpose(attend_w, [0, 2, 1]) + (-1 / attend_mask_w_1 + 1))
    attend_1 = tf.matmul(attend_norm_1, emb_[1])
    attend_2 = tf.matmul(attend_norm_2, emb_[0])

    l_compare = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_compare], [])
    compare_1 = apply_layers(l_compare, tf.concat([emb_[0], attend_1], 2), training=training)
    compare_2 = apply_layers(l_compare, tf.concat([emb_[1], attend_2], 2), training=training)

    agg_1 = tf.reduce_sum(tf.reshape(mask_1, [batch_size_, -1, 1]) * compare_1, 1)
    agg_2 = tf.reduce_sum(tf.reshape(mask_2, [batch_size_, -1, 1]) * compare_2, 1)
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
        sess.run(tf.global_variables_initializer())

        # start sampling
        q_train, q_valid, q_test = Queue(1), Queue(1), Queue(1)
        Process(target=sample, args=(train, word_to_index, emb_unknown, epoch_size, batch_size, q_train)).start()
        Process(target=sample, args=(val, word_to_index, emb_unknown, epoch_size, batch_size, q_valid)).start()
        Process(target=sample, args=(test, word_to_index, emb_unknown, epoch_size, batch_size, q_test)).start()

        # load pretrained word embeddings
        emb_0 = tf.Variable(0., validate_shape=False)
        saver = tf.train.Saver({'emb': emb_0})
        saver.restore(sess, '__cache__/tf/emb/model.ckpt')

        # embedding transforms
        if emb_center or emb_proj_pca:
            sess.run(emb_0.assign(emb_0 - tf.reshape(tf.reduce_mean(emb_0, axis=1), [-1, 1])))
        if emb_proj_pca:
            if emb_normalize:
                sess.run(emb_0.assign(emb_0 / tf.reshape(tf.norm(emb_0, axis=1), [-1, 1])))
            sess.run(l_proj_emb.kernel.assign(tf.svd(emb_0)[2][:, :200]))
        if emb_normalize:
            sess.run(emb[:tf.shape(emb_0)[0]].assign(emb_0))
            sess.run(emb.assign(emb / tf.reshape(tf.norm(emb, axis=1), [-1, 1])))

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            total_loss, val_correct, test_correct = 0, 0, 0
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in q_train.get():
                _, batch_loss = sess.run([train_op, loss], feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, y: y_, training: True,
                })
                total_loss += len(y_) * batch_loss
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in q_valid.get():
                logits_ = sess.run(logits, feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, training: False,
                })
                val_correct += np.sum(np.argmax(logits_, 1) == y_)
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in q_test.get():
                logits_ = sess.run(logits, feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, training: False,
                })
                test_correct += np.sum(np.argmax(logits_, 1) == y_)
            print(
                datetime.datetime.now(),
                f'finished epoch {i}, loss: {total_loss / len(train):f}, '
                f'val acc: {val_correct / len(val):f}, test acc: {test_correct / len(test):f}'
            )


def main():
    train, val, test = load_data()
    word_to_index = gen_tables(train)
    run_model(
        train, val, test, word_to_index=word_to_index, intra_sent=True, emb_unknown=100, emb_size=300,
        emb_center=True, emb_proj=200, emb_proj_pca=True, emb_normalize=True,
        n_intra=[200, 200], n_intra_bias=10, n_attend=[200, 200], n_compare=[200, 200], n_classif=[200, 200],
        dropout_rate=0.5, lr=0.001, batch_size=512, epoch_size=400
    )

if __name__ == '__main__':
    main()
