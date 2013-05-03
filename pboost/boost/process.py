import numpy as np
from bitarray import bitarray
from mpi4py import MPI
from pboost.boost.confidence_rated import ConfidenceRatedBoosting,ConfidenceRatedWL
from pboost.boost.adaboost import AdaBoost, AdaBoostWL
from pboost.boost.adaboost_fast import AdaBoostFast, AdaBoostFastWL

class Process():
    def __init__(self, pb, xval_ind, classifyEN=True):
        """
        
        Set up the environment for a boosting process
        
        Parameters
        ----------
        
        pb : pboost.environment.pb object
            Contains data related to whole program
            
        xval_ind : integer
            Cross validation index
        
        classifyEN : boolean, optional
            Flag to enable predictions
            
        """
        self.xval_ind = xval_ind
        self.pb = pb
        self.classifyEN = classifyEN
        self.isXvalMain = xval_ind == 0  # Training on the whole dataset

        self.out_fp = str(pb.wd + "out_" + str(pb.conf_num) + 
                                  "_" + str(xval_ind) + ".npz")

        self.label = pb.get_label(source = 'train')
        self.train_indices = (pb.xval_indices != xval_ind)
        self.train_label = self.label[self.train_indices]
        self.train_exam_no = sum(self.train_indices)
        self.val_indices = None
        self.val_label = None
        self.val_exam_no = None
        
        if self.pb.xvalEN and not self.isXvalMain:
            self.val_indices = (pb.xval_indices == xval_ind)
            self.val_label = self.label[self.val_indices]
            self.val_exam_no = sum(self.val_indices)
    
    def run(self):
        """
        
        Run the boosting process with given parameters.
        
        """
        if self.pb.xvalEN and not self.isXvalMain:
            dt = np.ones(self.pb.total_exam_no,
                         dtype="float32") / self.train_exam_no
            dt[self.val_indices] = 0.0
        else:
            dt = np.ones(self.pb.total_exam_no,
                         dtype="float32") / self.pb.total_exam_no

        val = np.zeros(8,dtype="float32")-1
        boosting = None
        wl = None
        if self.pb.algorithm == 'conf-rated':
            boosting = ConfidenceRatedBoosting(self)
            wl = ConfidenceRatedWL(self)
        elif self.pb.algorithm == 'adaboost':
            boosting = AdaBoost(self)
            wl = AdaBoostWL(self)
        elif self.pb.algorithm == 'adaboost-fast':
            boosting = AdaBoostFast(self)
            wl = AdaBoostFastWL(self)
        else:
            raise Exception("Unknown Boosting Algorithm")
        
        for r in range(self.pb.rounds):
            tree = wl.run(dt)
            dt = boosting.run(dt = dt,
                              r = r,
                              tree = tree)
            
        if self.isXvalMain:
            boosting.finalize()
        
        """Sync the predictions and save them to a file"""
        if self.pb.isLeader:
            if self.pb.xvalEN and not self.isXvalMain:
                val_predictions = boosting.get_val_predictions()
                hypotheses = boosting.get_hypotheses()
                np.savez(self.out_fp,
                         val_predictions = val_predictions,
                         hypotheses = hypotheses,
                         )
            if self.pb.testEN and self.isXvalMain:
                train_predictions = np.zeros([self.pb.total_exam_no,self.pb.rounds],
                                                  dtype="float32")
                test_predictions = np.zeros([self.pb.test_exam_no,self.pb.rounds],
                                                  dtype="float32")
                for slv in np.arange(self.pb.comm_size):
                    tr_i1 = self.pb.train_partition[slv]
                    tr_i2 = self.pb.train_partition[slv+1]
                    te_i1 = self.pb.test_partition[slv]
                    te_i2 = self.pb.test_partition[slv+1]
                    if slv == 0:
                        train_predictions[tr_i1:tr_i2,:] = boosting.get_train_predictions()
                        test_predictions[te_i1:te_i2,:] = boosting.get_test_predictions()
                    else:
                        train_predictions[tr_i1:tr_i2,:] = self.pb.comm.recv(source=slv,tag=11)
                        test_predictions[te_i1:te_i2,:] = self.pb.comm.recv(source=slv,tag=12)
                hypotheses = boosting.get_hypotheses()
                np.savez(self.out_fp,
                         train_predictions = train_predictions,
                         test_predictions = test_predictions,
                         hypotheses = hypotheses,
                         )
            if not self.pb.testEN and self.isXvalMain:
                train_predictions = np.zeros([self.pb.total_exam_no,self.pb.rounds],
                                                  dtype="float32")
                for slv in np.arange(self.pb.comm_size):
                    tr_i1 = self.pb.train_partition[slv]
                    tr_i2 = self.pb.train_partition[slv+1]
                    if slv == 0:
                        train_predictions[tr_i1:tr_i2,:] = boosting.get_train_predictions()
                    else:
                        train_predictions[tr_i1:tr_i2,:] = self.pb.comm.recv(source=slv,tag=11)
                hypotheses = boosting.get_hypotheses()
                np.savez(self.out_fp,
                         train_predictions = train_predictions,
                         hypotheses = hypotheses,
                         )
        else:
            if self.pb.testEN and self.isXvalMain:
                train_predictions = boosting.get_train_predictions()
                self.pb.comm.send(train_predictions,dest = 0,tag=11)
                test_predictions = boosting.get_test_predictions()
                self.pb.comm.send(test_predictions,dest = 0,tag=12)
            if not self.pb.testEN and self.isXvalMain:
                train_predictions = boosting.get_train_predictions()
                self.pb.comm.send(train_predictions,dest = 0,tag=11)