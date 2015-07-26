import h5py
import sys
import os
import glob
import time
import numpy as np
from pboost.boost.confidence_rated import ConfidenceRated, ConfidenceRatedWL
from pboost.boost.adaboost import AdaBoost, AdaBoostWL
from pboost.boost.adaboost_fast import AdaBoostFast, AdaBoostFastWL, AdaBoostFastWLMPI
from pboost.boost.rankboost import RankBoost


class PBoostLite():

    def __init__(self,
                 algorithm='adaboost-fast',
                 rounds=100,
                 omp_threads=4,
                 tree_depth=1,
                 conf_num=0,
                 wd='./'):
        self.label = None
        self.train_exam_no = None
        self.algorithm = algorithm
        self.rounds = rounds
        self.classifyEN = True
        self.isXvalMain = True
        self.pb = type('Process', (object,), {})
        self.pb.rounds = rounds
        self.pb.depth = tree_depth
        self.pb.omp_threads = omp_threads
        self.pb.rank = 0
        self.pb.testEN = False
        self.pb.train_ind2 = None
        self.pb.train_ind1 = 0
        self.pb.wd = wd
        self.pb.conf_num = conf_num
        self.pb.index_matrix = None
        self.pb.total_exam_no = None
        self.pb.feature_mapping = None
        self.fitted = None

    def fit(self, X, y):
        self.label = y
        self.pb.index_matrix = np.argsort(X, axis=0).T
        self.pb.feature_mapping = np.arange(X.shape[1])
        self.pb.total_exam_no = len(y)
        self.train_exam_no = len(y)
        self.pb.train_ind2 = self.train_exam_no
        dt = np.ones(self.train_exam_no, dtype="float32") / self.train_exam_no

        val = np.zeros(8, dtype="float32") - 1
        boosting = None
        wl = None
        if self.algorithm == 'conf-rated':
            boosting = ConfidenceRated(self)
            wl = ConfidenceRatedWL(self)
        elif self.algorithm == 'adaboost':
            boosting = AdaBoost(self)
            wl = AdaBoostWL(self)
        elif self.algorithm == 'adaboost-fast':
            boosting = AdaBoostFast(self)
            wl = AdaBoostFastWL(self)
        elif self.algorithm == 'rankboost':
            boosting = RankBoost(self)
            wl = ConfidenceRatedWL(self)
        elif self.algorithm == 'rankboost-fast':
            boosting = RankBoost(self)
            wl = AdaBoostFastWL(self)
        else:
            raise Exception("Unknown Boosting Algorithm")

        for r in range(self.rounds):
            tree = wl.run(dt)
            dt = boosting.run(dt=dt,
                              r=r,
                              tree=tree)
        inverse = dict()
        rnk_list = list()
        for r in np.arange(self.rounds):
            h = boosting.hypotheses[r]
            for node in h.get_all_nodes():
                rnk = node.rnk
                try:
                    inverse[rnk].append(node)
                except KeyError:
                    inverse[rnk] = [node, ]
                    rnk_list.append(rnk)
                    pass

        for rnk in rnk_list:
            unsorted_ds = X.T
            for node in inverse[rnk]:
                d1 = node.d1
                d3 = node.d3
                d4 = node.d4
                vec = unsorted_ds[d1, :]
                node.v = (vec[d3] + vec[d4]) / 2.0
        self.fitted = boosting
        return self.fitted.train_predictions[:, -1]


class PBoostLiteXval():

    def __init__(self,
                 algorithm='adaboost-fast',
                 rounds=100,
                 omp_threads=4,
                 tree_depth=1,
                 conf_num=0,
                 wd='./',
                 xval_no=10):
        self.label = None
        self.train_exam_no = None
        self.algorithm = algorithm
        self.rounds = rounds
        self.classifyEN = True
        self.isXvalMain = False
        self.pb = type('Process', (object,), {})
        self.pb.rounds = rounds
        self.pb.depth = tree_depth
        self.pb.omp_threads = omp_threads
        self.pb.rank = 0
        self.pb.testEN = False
        self.pb.train_ind2 = None
        self.pb.train_ind1 = 0
        self.pb.wd = wd
        self.pb.conf_num = conf_num
        self.pb.index_matrix = None
        self.pb.total_exam_no = None
        self.pb.feature_mapping = None
        self.fitted = None
        self.xval_no = xval_no
        self.val_exam_no = None
        self.val_indices = None
        self.pb.xvalEN = True
        self.pb.isLeader = True

    def fit(self, X, y, indices=None):
        self.label = y
        self.pb.index_matrix = np.argsort(X, axis=0).T
        self.pb.feature_mapping = np.arange(X.shape[1])
        self.pb.total_exam_no = len(y)
        if indices == None:
            indices = np.repeat(np.arange(1, (self.xval_no + 1)),
                                self.pb.total_exam_no / self.xval_no)
            np.random.shuffle(indices)

        val_predictions = np.zeros([self.pb.total_exam_no, self.rounds])

        for xval_ind in np.arange(1, self.xval_no + 1):
            self.val_indices = (indices == xval_ind)
            val_label = self.label[self.val_indices]
            self.val_exam_no = sum(self.val_indices)

            train_indices = (indices != xval_ind)
            train_label = self.label[train_indices]
            self.train_exam_no = sum(train_indices)
            self.pb.train_ind2 = self.train_exam_no
            dt = np.ones(self.pb.total_exam_no,
                         dtype="float32") / self.train_exam_no
            dt[self.val_indices] = 0.0

            val = np.zeros(8, dtype="float32") - 1
            boosting = None
            wl = None
            if self.algorithm == 'conf-rated':
                boosting = ConfidenceRated(self)
                wl = ConfidenceRatedWL(self)
            elif self.algorithm == 'adaboost':
                boosting = AdaBoost(self)
                wl = AdaBoostWL(self)
            elif self.algorithm == 'adaboost-fast':
                boosting = AdaBoostFast(self)
                wl = AdaBoostFastWL(self)
            elif self.algorithm == 'rankboost':
                boosting = RankBoost(self)
                wl = ConfidenceRatedWL(self)
            elif self.algorithm == 'rankboost-fast':
                boosting = RankBoost(self)
                wl = AdaBoostFastWL(self)
            else:
                raise Exception("Unknown Boosting Algorithm")

            for r in range(self.rounds):
                tree = wl.run(dt)
                dt = boosting.run(dt=dt,
                                  r=r,
                                  tree=tree)
                val_predictions[self.val_indices, :] = boosting.val_predictions

        return val_predictions
