#切分出两个验证集
#使用nfetc计算若干步，拿到corruption_matrix
#先聚类，
#然后搞到每个Cluster的cm：ccm
import copy
import datetime
import json
import logging
import random

from cluster_utils import feature_cluster, reconsider_cluster, reconsider_cluster_need_walk, reconsider_cluster2, \
    reconsider_cluster3, reconsider_cluster4
from utils import data_utils, embedding_utils, pkl_utils
from utils.eval_utils import strict, loose_macro, loose_micro, label_path
import numpy as np
from sklearn.model_selection import ShuffleSplit
import os
import config
import pickle
import tensorflow as tf
from nfetc_cglc import NFETC_AR

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Task:
    def __init__(self, model_name, data_name, cv_runs, params_dict, logger, portion=100, save_name='', re_init=False):
        print('Loading data...')
        if portion <= 100:  # all the data, portion% clean + all noisy
            self.portion = '-' + str(portion) if portion != 100 else ''
        else:
            portion /= 100  # only clean data, portion% clean
            self.portion = '-' + str(int(portion)) + '-clean'
        print('run task on: ', self.portion, ' dataset: ', data_name)
        self.data_name = data_name

        if data_name == 'ontonotes':
            words_train, mentions_train, positions_train, labels_train = data_utils.load(
                config.ONTONOTES_TRAIN_CLEAN + self.portion)
            words, mentions, positions, labels = data_utils.load(config.ONTONOTES_TEST_CLEAN)
            words_gt, mentions_gt, positions_gt, labels_gt = data_utils.load(config.ONTONOTES_GT_CLEAN)
            type2id, typeDict = pkl_utils.load(config.ONTONOTES_TYPE)
            num_types = len(type2id)
            type_info = config.ONTONOTES_TYPE

        elif data_name == 'bbn':
            words_train, mentions_train, positions_train, labels_train = data_utils.load(
                config.BBN_TRAIN_CLEAN + self.portion)
            # print(labels_train)
            words, mentions, positions, labels = data_utils.load(config.BBN_TEST_CLEAN)
            type2id, typeDict = pkl_utils.load(config.BBN_TYPE)
            num_types = len(type2id)
            type_info = config.BBN_TYPE

        elif data_name == 'wikim':
            words_train, mentions_train, positions_train, labels_train = data_utils.load(config.WIKIM_TRAIN_CLEAN)
            words_gt, mentions_gt, positions_gt, labels_gt = data_utils.load(config.WIKIM_GT_CLEAN)
            words, mentions, positions, labels = data_utils.load(config.WIKIM_TEST_CLEAN)
            type2id, typeDict = pkl_utils.load(config.WIKIM_TYPE)
            num_types = len(type2id)
            type_info = config.WIKIM_TYPE
        else:
            assert False, 'you have to specify the name of dataset with -d (ie. bbn/....)'

        self.model_name = model_name
        self.savename = save_name
        self.data_name = data_name
        self.cv_runs = cv_runs
        self.num_classes = len(type2id)
        self.params_dict = params_dict
        self.hparams = AttrDict(params_dict)
        self.logger = logger
        self.batch_size = self.hparams.batch_size
        self.num_epochs = self.hparams.num_epochs

        self.pseudo_gt = self.params_dict['pseudo_label']
        self.exp4 = self.params_dict['exp4']
        self.aux_clusters = self.hparams.aux_clusters
        self.e_1 = self.hparams.e_1
        self.e_2 = self.hparams.e_2
        self.e_3 = self.hparams.e_3

        self.re_init = re_init
        self.gt_portion = self.hparams.gt_portion

        self.id2type = {type2id[x]: x for x in type2id.keys()}
        self.type2id = type2id

        def type2vec(types):  # only terminal will be labeled
            tmp = np.zeros(num_types)
            for t in str(types).split():
                if t in type2id.keys():
                    tmp[type2id[t]] = 1.0
            return tmp

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        labels_train_ori = labels_train
        labels_train = np.array([type2vec(t) for t in labels_train])  # one hot vec'
        labels_train_s = np.array([softmax(x) for x in labels_train])

        train_id = np.array(range(len(labels_train))).astype(np.int32)

        tempname = self.data_name + config.testemb
        tempname = os.path.join(config.PKL_DIR, tempname)
        if os.path.exists(tempname):
            self.embedding = pickle.load(open(tempname, 'rb'))
            print('embedding load over')
        else:
            self.embedding = embedding_utils. \
                Embedding.fromCorpus(config.EMBEDDING_DATA, list(words_train) + list(words),
                                     config.MAX_DOCUMENT_LENGTH, config.MENTION_SIZE)
            pickle.dump(self.embedding, open(tempname, 'wb'))
            print('embedding dump over')
        self.embedding.max_document_length = config.MAX_DOCUMENT_LENGTH

        print('Preprocessing data...')

        if True:
            textlen_train = np.array(
                [self.embedding.len_transform1(x) for x in words_train])  # with cut down len sequence
            words_train = np.array([self.embedding.text_transform1(x) for x in
                                    words_train])  # with cut down word id sequence and mask with zero <PAD>
            mentionlen_train = np.array([self.embedding.len_transform2(x) for x in mentions_train])  # mention len
            mentions_train = np.array(
                [self.embedding.text_transform2(x) for x in mentions_train])  # mention text indexer
            positions_train = np.array(
                [self.embedding.position_transform(x) for x in positions_train])  # start ,end position
            print('get train data')

            textlen = np.array([self.embedding.len_transform1(x) for x in words])
            words = np.array([self.embedding.text_transform1(x) for x in words])  # padding and cut down
            mentionlen = np.array([self.embedding.len_transform2(x) for x in mentions])
            mentions = np.array([self.embedding.text_transform2(x) for x in mentions])
            positions = np.array([self.embedding.position_transform(x) for x in positions])
            print('get test data')

        labels_ori = labels
        labels = np.array([type2vec(t) for t in labels])

        #test/valid(+gt?)
        ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=config.RANDOM_SEED)
        for test_index, keep_index in ss.split(np.zeros(len(labels)), labels):#[train_size, test_size]
            textlen_test, textlen_keep = textlen[test_index], textlen[keep_index]
            words_test, words_keep = words[test_index], words[keep_index]
            mentionlen_test, mentionlen_keep = mentionlen[test_index], mentionlen[keep_index]
            mentions_test, mentions_keep = mentions[test_index], mentions[keep_index]
            positions_test, positions_keep = positions[test_index], positions[keep_index]
            labels_test, labels_keep = labels[test_index], labels[keep_index]
            labels_test_ori = labels_ori[test_index]
            labels_keep_ori = labels_ori[keep_index]

        if self.data_name == 'bbn':  # bbn
            ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=config.RANDOM_SEED)

            for gt_index, valid_index in ss.split(np.zeros(len(labels_keep)), labels_keep):#[train_size, gt_size]
                textlen_gt, textlen_valid = textlen_keep[gt_index], textlen_keep[valid_index]
                words_gt, words_valid = words_keep[gt_index], words_keep[valid_index]
                mentionlen_gt, mentionlen_valid = mentionlen_keep[gt_index], mentionlen_keep[valid_index]
                mentions_gt, mentions_valid = mentions_keep[gt_index], mentions_keep[valid_index]
                positions_gt, positions_valid = positions_keep[gt_index], positions_keep[valid_index]
                labels_gt, labels_valid = labels_keep[gt_index], labels_keep[valid_index]
                labels_gt_ori = labels_keep_ori[gt_index]
                labels_valid_ori = labels_keep_ori[valid_index]
        else:
            #valid
            textlen_valid = textlen_keep
            words_valid = words_keep
            mentionlen_valid = mentionlen_keep
            mentions_valid = mentions_keep
            positions_valid = positions_keep
            labels_valid = labels_keep
            labels_valid_ori = labels_keep_ori

            #gt
            textlen_gt = np.array([self.embedding.len_transform1(x) for x in words_gt])
            words_gt = np.array([self.embedding.text_transform1(x) for x in words_gt])
            mentionlen_gt = np.array([self.embedding.len_transform2(x) for x in mentions_gt])
            mentions_gt = np.array([self.embedding.text_transform2(x) for x in mentions_gt])
            positions_gt = np.array([self.embedding.position_transform(x) for x in positions_gt])
            labels_gt_ori = labels_gt
            labels_gt = np.array([type2vec(t) for t in labels_gt])

        self.train_set = list(
            zip(words_train, textlen_train, mentions_train, mentionlen_train, positions_train, labels_train, train_id))
        self.valid_set = list(
            zip(words_valid, textlen_valid, mentions_valid, mentionlen_valid, positions_valid, labels_valid, ))
        self.test_set = list(
            zip(words_test, textlen_test, mentions_test, mentionlen_test, positions_test, labels_test, ))
        self.gt_set = list(
            zip(words_gt, textlen_gt, mentions_gt, mentionlen_gt, positions_gt, labels_gt, ))

        print(labels)
        print(labels_train)
        # exit()

        self.full_test_set = list(zip(words, textlen, mentions, mentionlen, positions, labels, ))

        self.labels_train_ori = labels_train_ori
        self.labels_ori = labels_ori
        self.labels_test_ori = labels_test_ori
        self.labels_valid_ori = labels_valid_ori
        self.labels_gt_ori = labels_gt_ori

        self.labels_train = labels_train
        self.labels_test = labels_test
        self.labels = labels
        self.labels_train_s = labels_train_s
        self.labels_valid = labels_valid

        self.num_types = num_types
        self.num_train = len(labels_train)
        self.type_info = type_info
        self.logger.info('train set size:%d, test set size: %d' % (len(self.train_set), len(self.full_test_set)))

        self.model = self._get_model()
        self.saver = tf.train.Saver(tf.global_variables())
        checkpoint_dir = os.path.abspath(config.CHECKPOINT_DIR)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.checkpoint_prefix = os.path.join(checkpoint_dir, self.__str__())
        self.gt_set_full = copy.deepcopy(self.gt_set)
        self.labels_gt_ori_full = copy.deepcopy(self.labels_gt_ori)

    def __str__(self):
        return self.model_name + self.savename

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _get_model(self):
        np.random.seed(config.RANDOM_SEED)
        kwargs = {
            'sequence_length': config.MAX_DOCUMENT_LENGTH,
            'mention_length': config.MENTION_SIZE,
            'num_classes': self.num_types,
            'vocab_size': self.embedding.vocab_size,
            'embedding_size': self.embedding.embedding_dim,
            'position_size': self.embedding.position_size,
            'pretrained_embedding': self.embedding.embedding,
            'wpe': np.random.random_sample((self.embedding.position_size, self.hparams.wpe_dim)),
            'type_info': self.type_info,
            'num_train': self.num_train,
            'hparams': self.hparams,
            'dataset': self.data_name
        }
        return NFETC_AR(**kwargs)

    def _print_param_dict(self, d, prefix='      ', incr_prefix='      '):
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info('%s%s:' % (prefix, k))
                self.print_param_dict(v, prefix + incr_prefix, incr_prefix)
            else:
                self.logger.info('%s%s: %s' % (prefix, k, v))

    def create_session(self):
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=8,
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        return tf.Session(config=session_conf)

    def get_scores(self, preds, target='test', probs=None):
        preds = [label_path(self.id2type[x]) for x in preds]

        # print(self.test_set[0])
        def vec2type(v):
            s = []
            for i in range(len(v)):
                if v[i]:
                    s.extend(label_path(self.id2type[i]))
            return set(s)

        print('eval on ', target)
        if target == 'test':
            labels_test = [vec2type(x) for x in self.labels_test]  # path will caculate the father node for strict acc
        else:
            labels_test = [vec2type(x) for x in self.labels_valid]
        words = [self.embedding.i2w(k[0]) for k in self.full_test_set]
        mentions = [self.embedding.i2w(k[2]) for k in self.full_test_set]
        #oridata 没用到！
        acc = strict(labels_test, preds, oridata=(words, mentions), modelname=self.savename, probs=probs, logger=self.logger)
        _, _, macro = loose_macro(labels_test, preds)
        _, _, micro = loose_micro(labels_test, preds)
        return acc, macro, micro

    def filter_gt_samples(self):
        random_idx = np.random.permutation(len(self.gt_set_full))

        if self.gt_portion < 1:
            assert 0 < self.gt_portion
            random_idx = random_idx[:int(self.gt_portion * len(self.gt_set_full))]
            assert len(random_idx) > 0
            self.gt_set = [self.gt_set_full[i] for i in random_idx]
            self.labels_gt_ori = [self.labels_gt_ori_full[i] for i in random_idx]
        self.logger.info(f"GT samples: {len(self.gt_set)}.")

    def select_pseudo_gt_samples(self, sess):
        assert self.gt_portion == 1.0

        def pred2label(tp):
            tmp = np.zeros_like(self.gt_set_full[0][-1])
            tmp[tp] = 1
            return tmp

        preds, maxtype = self.model.predict_proba(sess, self.gt_set_full)
        threshold = self.pseudo_gt
        labels = []
        keep = []
        while len(labels) < 100 and threshold > -1:
            labels = [pred2label(tp) for pred, tp in zip(preds, maxtype) if max(pred) > threshold]
            keep = [1 if max(pred) > threshold else 0 for pred in preds]
            self.logger.info('pseudo label threshold  %.2f, %d samples now' % (threshold, len(labels)))
            threshold -= 0.03

        self.gt_set = [list(gt) for gt, kk in zip(self.gt_set_full, keep) if kk]
        # 把gt full也缩减为按照threshold筛选的gt set，并且用pseudo label代替label
        self.labels_gt_ori = [self.id2type[tp] for tp, kk in zip(maxtype, keep) if kk]
        for i in range(len(self.gt_set)):
            self.gt_set[i][-1] = labels[i]
            self.gt_set[i] = tuple(self.gt_set[i])
        assert len(self.gt_set) > 50

    def get_confusion_matrix(self, sess, epoch = 0):
        self.filter_gt_samples()
        # rst = 0
        print('num_classes', self.num_classes)

        if self.pseudo_gt < 1.0:
            self.select_pseudo_gt_samples(sess)
        preds, maxtype = self.model.predict_proba(sess, self.gt_set)
        # preds, maxtype = self.model.predict(sess, self.full_test_set)
        labels_gt_ori = self.labels_gt_ori
        # labels_gt_ori = self.labels_ori

        hoos, hoopl, hoolabels = self.model.get_h_output(sess, self.gt_set, mode='test')#使用验证集
        labels_gt_ori = self.labels_gt_ori
        gt_set_ori = self.gt_set
        num = len(gt_set_ori)
        print('{} gt samples'.format(num))

        hoos_train, hoopl_train, hoolabels_train = self.model.get_h_output(sess, self.train_set, mode='train')
        labels_train_ori = self.labels_train_ori
        t_set_train_ori = self.train_set
        num_train = len(t_set_train_ori)

        sampled_hoos = np.vstack((hoos, hoos_train))#注意test在前

        # aux_clusters = 50
        # aux_clusters = 20
        aux_clusters = self.aux_clusters

        print('start KNN for {} samples in epoch {}.'.format(num+num_train, epoch))
        preds_ori = reconsider_cluster4(sampled_hoos, hoolabels, hoolabels_train, n_runs=5, batch_size=5000,
                                               logger=self.logger, aux_clusters=aux_clusters, is_train=True, thres=0.5, verbose=0)
        print('end KNN for {} samples in epoch {}.'.format(num+num_train, epoch))
        clt_gt = preds_ori[:num]
        clt_train = preds_ori[num:]

        num_classes = self.num_classes
        num_clusters = self.num_classes + aux_clusters
        corruption_matrix = np.zeros((num_clusters, num_classes, num_classes))
        label_count = np.zeros((num_clusters, num_classes))
        cluster_count = np.zeros((num_clusters))
        for i, g_label in enumerate(labels_gt_ori):
            ori_types = g_label.split()
            cluster_id = clt_gt[i]
            cluster_count[cluster_id] += 1
            for t in ori_types:
                # print(i, t, preds[i])
                g_id = self.type2id[t]
                print(cluster_id, g_id)
                corruption_matrix[cluster_id, g_id] += preds[i]
                label_count[cluster_id, g_id] += 1

        for k in range(num_clusters):
            for i in range(num_classes):
                if label_count[k, i] == 0:
                    label_count[k, i] += 1
                    corruption_matrix[k,i, i] += 1
        for k in range(num_clusters):
            corruption_matrix[k] = corruption_matrix[k] / label_count[k, :, np.newaxis]

        cm_list = [corruption_matrix[i] for i in clt_train]
        cluster_count = cluster_count/(num/num_clusters)
        ct_list = [cluster_count[i] for i in clt_train]
        return cm_list, ct_list

    def refit(self):
        self.logger.info('Params')
        self._print_param_dict(self.params_dict)
        self.logger.info('Evaluation for each epoch')
        self.logger.info('\t\tEpoch\t\tAcc\t\tMacro\t\tMicro\t\tTAacc\t\tTMacro\t\tTMicro')
        # sess = self.create_session()

        print('retraining times: ', self.cv_runs)
        # sess.run(tf.global_variables_initializer())

        for para in tf.all_variables():
            print(para.name)

        maxbaseonvalid = ()

        vaacclist = []
        vamacrolist = []
        vamicrolist = []

        for i_cv in range(self.cv_runs):
            # if self.cv_runs > 1 and i_cv != 0:
            print(f'retraining times: {i_cv}/{self.cv_runs}' )
            print('Open sess_ori...')
                # sess.close()
            sess = self.create_session()
            sess.run(tf.global_variables_initializer())
            maxvaacc = -1
            # self.num_epochs = 3
            p1 = self.e_1
            p2 = self.e_2
            # for epoch in range(1, self.num_epochs+1):#预训练5次
            for epoch in range(1, p1):#预训练
                train_batches = data_utils.batch_iter(self.train_set, self.batch_size, 1)
                self.model.evaluate(sess, train_batches, is_correct=False, new_labels=[])

                preds, maxtype, probs = self.model.predict(sess, self.test_set)
                probs = probs if self.exp4 else None
                acc, macro, micro = self.get_scores(maxtype, probs=probs)
                vapreds, _, val_probs = self.model.predict(sess, self.valid_set)
                val_probs = val_probs if self.exp4 else None
                vaacc, vamacro, vamicro = self.get_scores(_, target='vatestset', probs=val_probs)
                cmp = vaacc
                if cmp >= maxvaacc:
                    maxvaacc = cmp
                    maxbaseonvalid = (epoch, acc, macro, micro, maxvaacc)
                    self.logger.info(
                        '\tepo\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' %
                        (epoch, vaacc, vamacro, vamicro, acc, macro, micro, maxvaacc))
                else:
                    self.logger.info(
                        '\tepo\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' %
                        (epoch, vaacc, vamacro, vamicro, acc, macro, micro))
            #计算混淆矩阵
            # cm, clt_train = self.get_confusion_matrix(sess)
            cm_list, ct_list = self.get_confusion_matrix(sess)
            if self.re_init:
                sess.close()
                print('Open sess_glc...')
                # sess.close()
                sess = self.create_session()
                sess.run(tf.global_variables_initializer())
            # for epoch in range(self.num_epochs, self.num_epochs*2):  # 预训练5次
            # for epoch in range(5, self.num_epochs*2):  # 预训练5次
            logging.info('Creating new train set for glc')
            words_, textlen_, mentions_, mentionlen_, positions_, labels_, ids_ = zip(*self.train_set)
            train_cglc = list(zip(words_, textlen_, mentions_, mentionlen_, positions_, labels_, ids_, cm_list, ct_list))
            words_, textlen_, mentions_, mentionlen_, positions_, labels_ = zip(*self.gt_set)
            train_cglc += list(zip(words_, textlen_, mentions_, mentionlen_, positions_, labels_, [0]*len(words_),
                                   [np.eye(self.num_classes)] * len(self.gt_set), [1.0] * len(self.gt_set)))
            random.shuffle(train_cglc)
            logging.info('Creating new train set for glc done')
            for epoch in range(p1, p2):  # 正式训练5次
                train_batches = data_utils.batch_iter(train_cglc, self.batch_size, 1)
                self.model.evaluate_cglc(sess, train_batches)
                preds, maxtype, probs = self.model.predict(sess, self.test_set)
                probs = probs if self.exp4 else None
                acc, macro, micro = self.get_scores(maxtype, probs=probs)
                vapreds, _, val_probs = self.model.predict(sess, self.valid_set)
                val_probs = val_probs if self.exp4 else None
                vaacc, vamacro, vamicro = self.get_scores(_, target='vatestset', probs=val_probs)
                cmp = vaacc
                if cmp >= maxvaacc:
                    maxvaacc = cmp
                    maxbaseonvalid = (epoch, acc, macro, micro, maxvaacc)
                    self.logger.info(
                        '\tepg\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' %
                        (epoch, vaacc, vamacro, vamicro, acc, macro, micro, maxvaacc))
                else:
                    self.logger.info(
                        '\tepg\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' %
                        (epoch, vaacc, vamacro, vamicro, acc, macro, micro))

            vaacclist.append(maxbaseonvalid[1])
            vamacrolist.append(maxbaseonvalid[2])
            vamicrolist.append(maxbaseonvalid[3])
            self.logger.info('\tMax\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' % (
                maxbaseonvalid[0], maxbaseonvalid[1], maxbaseonvalid[2], maxbaseonvalid[3], maxbaseonvalid[4]))

        meanvaacc = np.mean(vaacclist)
        meanvamacro = np.mean(vamacrolist)
        meanvamicro = np.mean(vamicrolist)
        stdvaacc = np.std(vaacclist)
        stdvamacro = np.std(vamacrolist)
        stdvamicro = np.std(vamicrolist)

        self.logger.info('\tCV\t{:.1f}±{:.1f}\t{:.1f}±{:.1f}\t{:.1f}±{:.1f}'.format(meanvaacc * 100, stdvaacc * 100,
                                                                                    meanvamacro * 100, stdvamacro * 100,
                                                                                    meanvamicro * 100,
                                                                                    stdvamicro * 100))
        sess.close()

        # self.refit_glc()

    def save(self, sess):
        path = self.saver.save(sess, self.checkpoint_prefix)
        self.embedding.save(self.checkpoint_prefix)
        print('Saved model to {}'.format(path))
        print('-' * 100)
