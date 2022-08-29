from model import Model
import tensorflow as tf
from utils import data_utils, prior_utils
from utils import eval_utils
import numpy as np
import config
from functools import reduce

tf.set_random_seed(seed=config.RANDOM_SEED)


class NFETC_AR(Model):
    def __init__(self, num_train, sequence_length, mention_length, num_classes, vocab_size,
                 embedding_size, position_size, pretrained_embedding, wpe, type_info, hparams, dataset):
        self.data_name = dataset
        self.sequence_length = sequence_length
        self.mention_length = mention_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.position_size = position_size
        self.pretrained_embedding = pretrained_embedding
        self.wpe = wpe
        self.num_train = num_train

        self.beta = hparams.beta
        self.gamma = hparams.gamma
        self.omega = hparams.omega
        self.delta = hparams.delta

        self.state_size = hparams.state_size
        self.hidden_layers = hparams.hidden_layers
        self.hidden_size = hparams.hidden_size
        self.wpe_dim = hparams.wpe_dim
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.lr = hparams.lr

        self.dense_keep_prob = hparams.dense_keep_prob
        self.dense_keep_prob2 = 0.25
        self.rnn_keep_prob = hparams.rnn_keep_prob

        self.hp = hparams
        self.batch_size = hparams.batch_size
        self.num_epochs = hparams.num_epochs

        # all one;no alpha
        self.prior = tf.Variable(prior_utils.create_prior(type_info), trainable=False, dtype=tf.float32, name='prior')
        self.tune = tf.Variable(np.transpose(prior_utils.create_prior(type_info, hparams.alpha)), trainable=False,
                                dtype=tf.float32, name='tune')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.build()

    def add_placeholders(self):
        self.input_words = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_words')
        self.input_textlen = tf.placeholder(tf.int32, [None], name='input_textlen')
        self.input_mentions = tf.placeholder(tf.int32, [None, self.mention_length], name='input_mentions')
        self.input_mentionlen = tf.placeholder(tf.int32, [None], name='input_mentionlen')
        self.input_positions = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_positions')
        self.input_labels = tf.placeholder(tf.float32, [None, self.num_classes], name='input_labels')
        self.cm = tf.placeholder(tf.float32, [None, self.num_classes, self.num_classes], name='cm')
        self.ct = tf.placeholder(tf.float32, [None], name='ct')
        # self.t_clt = tf.placeholder(tf.float32, [None, 1], name='t_clt')
        self.input_ids = tf.placeholder(tf.int32, [None], name='input_ids')
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.dense_dropout = tf.placeholder(tf.float32, name='dense_dropout')
        # self.dense_dropout2 = tf.placeholder(tf.float32, name='dense_dropout2')
        # self.dense_dropout2 = 1.
        # self.dense_dropout2 = tf.constant(0.3, tf.float32)
        self.rnn_dropout = tf.placeholder(tf.float32, name='rnn_dropout')

        tmp = [i for i in range(self.mention_length)]
        tmp[0] = self.mention_length
        interval = tf.Variable(tmp, trainable=False)
        interval_row = tf.expand_dims(interval, 0)
        upper = tf.expand_dims(self.input_mentionlen - 1, 1)
        mask = tf.less(interval_row, upper)
        self.mention = tf.where(mask, self.input_mentions, tf.zeros_like(self.input_mentions))
        self.mentionlen = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        self.mentionlen = tf.cast(
            tf.where(tf.not_equal(self.mentionlen, tf.zeros_like(self.mentionlen)), self.mentionlen,
                     tf.ones_like(self.mentionlen)), tf.float32)
        self.mentionlen = tf.expand_dims(self.mentionlen, 1)

    def create_feed_dict(self, input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
                         input_labels=None, input_ids=None, phase=False, dense_dropout=1., rnn_dropout=1., dense_dropout2 = 1, cm=None, ct=None):
        feed_dict = {
            self.input_words: input_words,
            self.input_textlen: input_textlen,
            self.input_mentions: input_mentions,
            self.input_mentionlen: input_mentionlen,
            self.input_positions: input_positions,
            self.phase: phase,
            self.dense_dropout: dense_dropout,
            # self.dense_dropout2: dense_dropout,
            # self.dense_dropout2: dense_dropout2,
            self.rnn_dropout: rnn_dropout
        }
        if input_labels is not None:
            feed_dict[self.input_labels] = input_labels
        if input_ids is not None:
            feed_dict[self.input_ids] = input_ids

        if cm is not None:
            feed_dict[self.cm] = cm

        if ct is not None:
            feed_dict[self.ct] = ct

        # if t_clt is not None:
        #     feed_dict[self.t_clt] = t_clt

        return feed_dict

    def add_embedding(self):
        with tf.device('/cpu:0'), tf.name_scope('word_embedding'):
            W = tf.Variable(self.pretrained_embedding, trainable=False, dtype=tf.float32, name='W')
            self.embedded_words = tf.nn.embedding_lookup(W, self.input_words)
            self.embedded_mentions = tf.nn.embedding_lookup(W, self.input_mentions)
            self.mention_embedding = tf.divide(tf.reduce_sum(tf.nn.embedding_lookup(W, self.mention),
                                                             axis=1), self.mentionlen)

        with tf.device('/cpu:0'), tf.name_scope('position_embedding'):
            W = tf.Variable(self.wpe, trainable=False, dtype=tf.float32, name='W')
            self.wpe_chars = tf.nn.embedding_lookup(W, self.input_positions)
        self.input_sentences = tf.concat([self.embedded_words, self.wpe_chars], 2)

    def add_hidden_layer(self, x, idx):
        dim = self.feature_dim if idx == 0 else self.hidden_size
        with tf.variable_scope('hidden_%d' % idx):
            W = tf.get_variable('W', shape=[dim, self.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            b = tf.get_variable('b', shape=[self.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            h = tf.nn.xw_plus_b(x, W, b)
            h_norm = tf.layers.batch_normalization(h, training=self.phase)
            h_drop = tf.nn.dropout(tf.nn.relu(h_norm), self.dense_dropout, seed=config.RANDOM_SEED, name='h_output')
            # h_drop = tf.nn.dropout(tf.nn.relu(h_norm), self.dense_dropout2, seed=config.RANDOM_SEED, name='h_output')
        return h_drop

    def extract_last_relevant(self, outputs, seq_len):
        batch_size = tf.shape(outputs)[0]
        max_length = int(outputs.get_shape()[1])
        num_units = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (seq_len - 1)
        flat = tf.reshape(outputs, [-1, num_units])
        relevant = tf.gather(flat, index)
        return relevant

    def add_prediction_op(self):
        self.add_embedding()
        self.bsize = tf.shape(self.embedded_mentions)[0]

        with tf.name_scope('sentence_repr'):
            attention_w = tf.get_variable('attention_w', [self.state_size, 1])
            cell_forward = tf.contrib.rnn.LSTMCell(self.state_size)
            cell_backward = tf.contrib.rnn.LSTMCell(self.state_size)
            cell_forward = tf.contrib.rnn.DropoutWrapper(cell_forward, input_keep_prob=self.dense_dropout,
                                                         output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)
            cell_backward = tf.contrib.rnn.DropoutWrapper(cell_backward, input_keep_prob=self.dense_dropout,
                                                          output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_forward, cell_backward, self.input_sentences,
                sequence_length=self.input_textlen, dtype=tf.float32)
            outputs_added = tf.nn.tanh(tf.add(outputs[0], outputs[1]))
            alpha = tf.nn.softmax(tf.reshape(tf.matmul(
                tf.reshape(outputs_added, [-1, self.state_size]),
                attention_w),
                [-1, self.sequence_length]))
            alpha = tf.expand_dims(alpha, 1)
            self.sen_repr = tf.squeeze(tf.matmul(alpha, outputs_added))

        with tf.name_scope('mention_repr'):
            cell = tf.contrib.rnn.LSTMCell(self.state_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dense_dropout,
                                                 output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)

            outputs, states = tf.nn.dynamic_rnn(
                cell, self.embedded_mentions,
                sequence_length=self.input_mentionlen, dtype=tf.float32)
            self.men_repr = self.extract_last_relevant(outputs, self.input_mentionlen)

        self.features = tf.concat([self.sen_repr, self.men_repr, self.mention_embedding], -1)
        # self.features = tf.concat([self.sen_repr, self.men_repr], -1)
        # self.features = self.sen_repr
        self.feature_dim = self.state_size * 2 + self.embedding_size
        # self.feature_dim = self.state_size * 2
        # self.feature_dim = self.state_size

        self.h_output_out = self.features
        # h_drop = tf.nn.dropout(tf.nn.relu(self.features), self.dense_dropout2, seed=config.RANDOM_SEED)
        h_drop = tf.nn.dropout(tf.nn.relu(self.features), self.dense_dropout, seed=config.RANDOM_SEED)
        h_drop.set_shape([None, self.feature_dim])
        h_output = tf.layers.batch_normalization(h_drop, training=self.phase, name='h_output')

        # tf.add_to_collection('h_output', h_output)

        # get representation layer
        for i in range(self.hidden_layers):
            h_output = self.add_hidden_layer(h_output, i)
        if self.hidden_layers == 0:
            self.hidden_size = self.feature_dim

        with tf.variable_scope('typeVec', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', shape=[self.hidden_size, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(
                                    seed=config.RANDOM_SEED))  # hidden size= 660
            b = tf.get_variable('b', shape=[self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            # self.h_output_out = h_output  # 取这个吗？？
            self.scores = tf.nn.xw_plus_b(h_output, W, b, name='scores')  # [batch,num class]
            self.proba = tf.nn.softmax(self.scores, name='proba')

            # self.h_output_out = self.scores #

            # hier
            self.adjusted_proba = tf.matmul(self.proba, self.tune)
            self.adjusted_proba = tf.clip_by_value(self.adjusted_proba, 1e-10, 1.0, name='adprob')

            # unleaked ori props
            self.maxtype = tf.argmax(self.proba, 1, name='maxtype')
            self.predictions = tf.one_hot(self.maxtype, self.num_classes, name='prediction')

    def add_loss_op(self):
        with tf.variable_scope('label_embeddings'):
            l_embeddings = tf.get_variable('l_embeddings', shape=[self.num_train, self.num_classes],#这是分布吧
                                           initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
            self.input_labels_v = tf.nn.embedding_lookup(l_embeddings, self.input_ids)
            self.input_labels_v = tf.clip_by_value(self.input_labels_v, 1e-10, 1.0)

        with tf.name_scope('loss'):
            target = tf.argmax(tf.multiply(self.adjusted_proba, self.input_labels), axis=1)
            target_index = tf.one_hot(target, self.num_classes)
            # on wikim dataset, follow calculation process of NFETC
            labels = self.input_labels if self.data_name != 'wikim' else target_index
            losses = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(self.adjusted_proba), 1))
            losses_c = tf.reduce_mean(#input_labels_v = p~ij; adjusted_proba = pj
                tf.reduce_sum(self.input_labels_v * (tf.log(self.input_labels_v) - tf.log(self.adjusted_proba)), 1))#kl
            losses_o = tf.reduce_mean(-tf.reduce_sum(self.input_labels * tf.log(self.input_labels_v), 1))#d
            losses_e = tf.reduce_mean(-tf.reduce_sum(self.adjusted_proba * tf.log(self.adjusted_proba), 1))#s
            self.l2_loss = tf.contrib.layers.apply_regularization(
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                weights_list=tf.trainable_variables())
            self.loss = losses + self.l2_loss
            self.loss_v = self.beta * losses + self.gamma * losses_c + self.omega * losses_o + self.delta * losses_e

            # mix_loss_s = -tf.reduce_sum(
            #     one_hot_labels * tf.cast(tf.math.logical_not(tf.cast(silver_gold_ids_, tf.bool)), tf.float32) * tf.log(
            #         probabilities_corrected), axis=-1)

            # self.probs_adjusted = tf.nn.softmax(tf.linalg.matmul(self.adjusted_proba, self.cm))
            # self.probs_adjusted = self.adjusted_proba tf.transpose(self.cm)
            # self.probs_adjusted = self.adjusted_proba * self.
            # self.probs_adjusted = tf.matmul(self.adjusted_proba, self.cm[self.t_clt, :, :])
            self.probs_adjusted = tf.matmul(tf.expand_dims(self.adjusted_proba,1),  self.cm)
            self.probs_adjusted = tf.squeeze(self.probs_adjusted, 1)

            # self.probs_adjusted = tf.matmul(self.adjusted_proba, tf.transpose(self.cm))
            # self.probs_adjusted = tf.nn.softmax(self.adjusted_proba)
            self.probs_adjusted = tf.clip_by_value(self.probs_adjusted, 1e-10, 10, name='probs_adjusted')
            # self.probs_adjusted = tf.div(self.probs_adjusted, tf.reduce_mean(self.probs_adjusted, -1, keepdims=True))
            # probs_adjusted = tf.linalg.matmul(self.adjusted_proba, tf.transpose(self.cm))

            losses_glc_ = -tf.reduce_sum(labels * tf.log(self.probs_adjusted), 1)
            # print(self.ct)
            losses_glc = tf.reduce_mean(losses_glc_ * self.ct)

            losses_ori = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(self.adjusted_proba), 1) * self.ct)
            # self.loss_glc = 0.1 * losses_glc + 0.9* losses + self.l2_loss
            aa = self.beta
            # temp_loss
            self.loss_glc = (1 - aa) * losses_glc + aa * losses_ori
            # self.loss_glc = losses_glc

    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.grads_and_vars_v = optimizer.compute_gradients(self.loss_v)
        self.grads_and_vars_glc = optimizer.compute_gradients(self.loss_glc)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op_v = optimizer.apply_gradients(self.grads_and_vars_v, global_step=self.global_step)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
            self.train_op_glc = optimizer.apply_gradients(self.grads_and_vars_glc, global_step=self.global_step)

    def train_on_batch(self, sess, input_words, input_textlen, input_mentions,
                       input_mentionlen, input_positions, input_labels, input_ids):
        feed = self.create_feed_dict(input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
                                     input_labels, input_ids, True, self.dense_keep_prob, self.rnn_keep_prob, self.dense_keep_prob2)
        # Variablelist = [self.train_op, self.global_step, self.loss, self.l2_loss, self.proba]
        Variablelist = [self.train_op, self.global_step, self.loss, self.l2_loss, self.proba]

        a = sess.run(Variablelist, feed_dict=feed)
        step = a[1]
        if step and step % 100 == 0:
            print('ORI step {}, loss {:g} l2_loss {:g}'.format(step, a[2], a[3]))
            # print(np.sum(a[-1], 1))

    def train_on_batch_glc(self, sess, input_words, input_textlen, input_mentions,
                       input_mentionlen, input_positions, input_labels, input_ids, cm):
        feed = self.create_feed_dict(input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
                                     input_labels, input_ids, True, self.dense_keep_prob, self.rnn_keep_prob, self.dense_keep_prob2, cm)
        # Variablelist = [self.train_op_glc, self.global_step, self.loss, self.l2_loss, self.cm, self.adjusted_proba, self.probs_adjusted]
        Variablelist = [self.train_op_glc, self.global_step, self.loss, self.l2_loss, self.cm, self.proba, self.probs_adjusted]
        # Variablelist = [self.train_op, self.global_step, self.loss, self.l2_loss, self.cm, self.adjusted_proba, self.probs_adjusted]

        a = sess.run(Variablelist, feed_dict=feed)
        step = a[1]
        if step and step % 100 == 0:
            print('GLC step {}, loss {:g} l2_loss {:g}'.format(step, a[2], a[3]))
            # print('probs_adjusted', a[-1])
            # print(np.round(np.sum(a[-1], 1), 2))
            print(np.sum(a[-3], 0))
            print(np.round(a[-3], 2))
            # exit(2)

    def train_on_batch_cglc(self, sess, input_words, input_textlen, input_mentions,
                       input_mentionlen, input_positions, input_labels, input_ids, cm, ct):
        feed = self.create_feed_dict(input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
                                     input_labels, input_ids, True, self.dense_keep_prob, self.rnn_keep_prob, self.dense_keep_prob2, cm, ct)
        # Variablelist = [self.train_op_glc, self.global_step, self.loss, self.l2_loss, self.cm, self.adjusted_proba, self.probs_adjusted]
        Variablelist = [self.train_op_glc, self.global_step, self.loss, self.l2_loss, self.cm, self.proba, self.probs_adjusted]
        # Variablelist = [self.train_op, self.global_step, self.loss, self.l2_loss, self.cm, self.adjusted_proba, self.probs_adjusted]

        a = sess.run(Variablelist, feed_dict=feed)
        step = a[1]
        if step and step % 100 == 0:
            print('GLC step {}, loss {:g} l2_loss {:g}'.format(step, a[2], a[3]))
            # print('probs_adjusted', a[-1])
            # print(np.round(np.sum(a[-1], 1), 2))
            # print(np.sum(a[-3], 0))
            # print(np.round(a[-3], 2))

    def train_on_batch_v(self, sess, input_words, input_textlen, input_mentions,
                         input_mentionlen, input_positions, input_labels, input_ids):
        feed = self.create_feed_dict(input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
                                     input_labels, input_ids, True, self.dense_keep_prob, self.rnn_keep_prob, self.dense_keep_prob2)
        Variablelist = [self.train_op_v, self.global_step, self.loss, self.l2_loss]

        a = sess.run(Variablelist, feed_dict=feed)
        step = a[1]
        if step and step % 100 == 0:
            print('step {}, loss {:g} l2_loss {:g}'.format(step, a[2], a[3]))

    def get_scores(self, preds, labels, id2type):
        label_path = eval_utils.label_path
        if type(preds) == np.ndarray:
            preds = [[label_path(id2type[i]) for i, x in enumerate(line) if x > 0] for line in preds]
            preds = [list(set(reduce(lambda x, y: x + y, line))) for line in preds]
        else:
            preds = [label_path(id2type[x]) for x in preds]

        def vec2type(v):
            s = []
            for i in range(len(v)):
                if v[i]:
                    s.extend(label_path(id2type[i]))
            return set(s)

        labels_test = [vec2type(x) for x in labels]  # path will caculate the father node for strict acc
        acc = eval_utils.strict(labels_test, preds)
        _, _, macro = eval_utils.loose_macro(labels_test, preds)
        _, _, micro = eval_utils.loose_micro(labels_test, preds)

        return acc, macro, micro

    def predict(self, sess, test):
        batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
        all_predictions = []
        all_probs = []
        all_labels = []
        all_maxtype = []
        for batch in batches:
            words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch = zip(*batch)

            feed = self.create_feed_dict(words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch)
            batch_predictions, batchmaxtype, batch_prob = sess.run([self.predictions, self.maxtype, self.proba], feed_dict=feed)
            if len(all_predictions) == 0:
                all_predictions = batch_predictions
            else:
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            if len(all_maxtype) == 0:
                all_maxtype = batchmaxtype
            else:
                all_maxtype = np.concatenate([all_maxtype, batchmaxtype])
            if len(all_probs) == 0:
                all_probs = batch_prob
            else:
                all_probs = np.concatenate([all_probs, batch_prob])

            if len(all_labels) == 0:
                all_labels = np.array(labels_batch)
            else:
                all_labels = np.concatenate([all_labels, np.array(labels_batch)])
        return all_predictions, all_maxtype, all_probs

    def predict_proba(self, sess, test):
        batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
        all_predictions = []
        all_labels = []
        all_maxtype = []
        for batch in batches:
            words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch = zip(*batch)

            feed = self.create_feed_dict(words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch)
            batch_predictions, batchmaxtype = sess.run([self.proba, self.maxtype], feed_dict=feed)
            if len(all_predictions) == 0:
                all_predictions = batch_predictions
            else:
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            if len(all_maxtype) == 0:
                all_maxtype = batchmaxtype
            else:
                all_maxtype = np.concatenate([all_maxtype, batchmaxtype])

            if len(all_labels) == 0:
                all_labels = np.array(labels_batch)
            else:
                all_labels = np.concatenate([all_labels, np.array(labels_batch)])
        return all_predictions, all_maxtype

    def get_h_output(self, sess, test, mode = 'test'):
        batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
        all_hoos = []
        all_labels = []
        all_maxtype = []
        for batch in batches:
            # words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, _ = zip(*batch)
            if mode == 'test':
                words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch = zip(*batch)
            else:
                words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, _ = zip(
                    *batch)

            feed = self.create_feed_dict(words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch)
            batch_hoos, batch_maxtype = sess.run([self.h_output_out, self.maxtype], feed_dict=feed)
            if len(all_hoos) == 0:
                all_hoos = batch_hoos
            else:
                all_hoos = np.concatenate([all_hoos, batch_hoos])

            if len(all_maxtype) == 0:
                all_maxtype = batch_maxtype
            else:
                all_maxtype = np.concatenate([all_maxtype, batch_maxtype])

            if len(all_labels) == 0:
                all_labels = np.array(labels_batch)
            else:
                all_labels = np.concatenate([all_labels, np.array(labels_batch)])

        return all_hoos, all_maxtype, all_labels


    def evaluate(self, sess, train_batches, is_correct=False, new_labels=None):
        if is_correct:#s2
            for batch in train_batches:
                words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, ids_batch = zip(
                    *batch)
                self.train_on_batch_v(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch,
                                      positions_batch, labels_batch, ids_batch)
        else:
            if len(new_labels) == 0:#s1
                for batch in train_batches:
                    words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, ids_batch = zip(
                        *batch)
                    self.train_on_batch(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch,
                                        positions_batch, labels_batch, ids_batch)
            else:
                for batch in train_batches:#s3
                    words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, ids_batch = zip(
                        *batch)
                    new_labels_batch = np.array([new_labels[i] for i in ids_batch])
                    self.train_on_batch(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch,
                                        positions_batch, new_labels_batch, ids_batch)

        # return self.

    def evaluate_glc(self, sess, train_batches, cm):

        for batch in train_batches:
            words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, ids_batch = zip(
                *batch)
            self.train_on_batch_glc(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch,
                                positions_batch, labels_batch, ids_batch, cm)

    def evaluate_cglc(self, sess, train_batches):

        for batch in train_batches:
            words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch, ids_batch, cm_batch, ct_batch = zip(
                *batch)
            self.train_on_batch_cglc(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch,
                                positions_batch, labels_batch, ids_batch, cm_batch, ct_batch)
