import h5py
import numpy as np
from numexpr import evaluate
from pboost.boost.generic import Boosting, WeakLearner

class AdaBoost(Boosting):
    def __init__(self, boosting_p):
        """
        
        AdaBoost Implementation
        
        boosting_p
            Boosting environment in which this algoritm runs
            
        """
        self.process = boosting_p
        self.pb = boosting_p.pb
        
        self.hypotheses = list()
        self.val_predictions = None
        self.train_predictions = None
        self.test_predictions = None

        if self.process.classifyEN:
            if self.process.isXvalMain:
                train_span = self.pb.train_ind2 - self.pb.train_ind1
                self.train_predictions = np.zeros(
                [train_span,self.pb.rounds],'float32')
                if self.pb.testEN:
                    test_span = self.pb.test_ind2 - self.pb.test_ind1
                    self.test_predictions = np.zeros(
                    [test_span,self.pb.rounds],'float32')
            else:
                if self.pb.isLeader and self.pb.xvalEN:
                    self.val_predictions = np.zeros(
                    [self.process.val_exam_no,self.pb.rounds],'float32')
        
        self.tl = np.int16(np.copy(self.process.label))
        self.tl[self.process.label == 0] = -1
        

    def run(self, dt, r, val, bout):
        """
        
        Run a single round of boosting
        
        dt
            Distribution over examples
        r
            Round number
        
        Returns:
            Updated distribution over examples

        """
        
        """Find the weak learner with the least weighted error""" 
        (rnk, d1, d2, d3, d4, c0, c1) = val[1:8]
        rnk = int(rnk)
        """Get the unsorted results for the hypothesis found by WL"""
        v = 0.0
                
        """Create a dictionary for the hypothesis found in this round"""
        h = {'rnk':rnk,
             'd1':d1,
             'd2':d2,
             'd3':d3,
             'd4':d4,
             'v':v,
             'c0':c0,
             'c1':c1}
        
        self.hypotheses.append(h)
        if self.process.classifyEN:
            if self.process.isXvalMain:
                pbout = bout[self.pb.train_ind1:self.pb.train_ind2]
                npbout = np.logical_not(pbout)
                s1 = pbout*c0
                s2 = npbout*c1
                self.train_predictions[:,r]=(self.train_predictions[:,r-1]
                                               + s1 + s2)
            else:
                if self.pb.isLeader and self.pb.xvalEN:
                    pbout = bout[self.process.val_indices]
                    npbout = np.logical_not(pbout)
                    s1 = pbout*c0
                    s2 = npbout*c1
                    self.val_predictions[:,r]=(self.val_predictions[:,r-1]
                                                   + s1 + s2)
        
        nbout = np.logical_not(bout)
        dt[bout] = dt[bout] * np.exp(-self.tl[bout] * np.float32(c0))
        dt[nbout] = dt[nbout] * np.exp(-self.tl[nbout]  * np.float32(c1))

        return dt / np.sum(dt)
    
    def finalize(self):
        """
        Write hypotheses and predictions into a file
        """
        inverse = dict()
        rnk_list = list()
        for r in np.arange(self.pb.rounds):
            h = self.hypotheses[r]
            rnk = h['rnk']
            try:
                inverse[rnk].append(r)
            except KeyError:
                inverse[rnk] = [r,]
                rnk_list.append(rnk)
                pass
        
        for rnk in rnk_list:
            model_fp = self.pb.wd + "model_%s_%s.h5" % (self.pb.conf_num,rnk)
            try:
                mf = h5py.File(model_fp,'r')
            except Exception as e:
                print model_fp
                print e
            unsorted_ds = mf["train_unsorted"]
            for r  in inverse[rnk]:
                h = self.hypotheses[r]
                d1 = h['d1']
                d3 = h['d3']
                d4 = h['d4']
                vec = unsorted_ds[d1, :]
                h['v'] = (vec[d3]+vec[d4])/2.0
                self.hypotheses[r] = h
                if self.pb.testEN:
                    tVals = mf["test_unsorted"][d1,self.pb.test_ind1:self.pb.test_ind2]
                    s1 = np.int16([tVals <= h['v']])*h['c0']
                    s2 = np.int16([tVals > h['v']])*h['c1']
                    self.test_predictions[:,r]= s1[0] + s2[0]
            mf.close()
        if self.pb.testEN:
            np.cumsum(self.test_predictions, axis=1, out=self.test_predictions)
        
        
    def get_hypotheses(self):
        """
        Return the hypotheses
        """
        return self.hypotheses
    
    def get_val_predictions(self):
        return self.val_predictions
    
    def get_train_predictions(self):
        return self.train_predictions
    
    def get_test_predictions(self):
        return self.test_predictions
        

class AdaBoostWL(WeakLearner):
    
    def __init__(self, boosting_p):
        """
        
        Decision Stump Implementation compatible with AdaBoost
        
        boosting_p
            Boosting environment in which this algoritm runs
            
        """
        self.process = boosting_p
        self.pb = boosting_p.pb

#        df = np.int32(np.cumsum(self.process.train_indices) * self.process.train_indices)
#        self.__cf = np.int32(np.cumsum(self.process.train_indices))
        
        self.__index = self.pb.index_matrix
        self.__w00 = np.zeros(shape = self.__index.shape,dtype ="float32")
        self.__w01 = np.zeros(shape = self.__index.shape,dtype ="float32")
        self.__err = np.zeros(shape = self.__index.shape,dtype ="float32")
        self.__bout = np.zeros(self.__index.shape[1],dtype="bool")
        self.__cdt = np.zeros(shape = self.process.label.shape,dtype ="float32")
        self.__cdnt = np.zeros(shape = self.process.label.shape,dtype ="float32")
    
    def run(self, dt):
        """
        
        Run a single round of weak learner
        
        dt
            Distribution over examples
        
        Returns:
            A list of WL related information
            err_best : The weighted error of the best hypothesis
            d1 : Index of the best hypothesis
            d2 : The relative index of the example where threshold is placed
            d3 : The actual index of the example where threshold is placed
            d4 : The actual index of the next example in threshold calculation
            c0 : Prediction for values lower than threshold
            c1 : Prediction for values larger than threshold
        """
        
        """Calculate the weighted error matrix for label=0 pred=0"""
        
        """Calculate the weighted error matrix for label=1 pred=0"""
        self.__cdnt = np.float32(dt * np.logical_not(self.process.label.T))
        self.__cdt = np.float32(dt * self.process.label.T)
        self.__w00 = np.take(self.__cdnt, self.__index)
        np.cumsum(self.__w00, axis=1, dtype ="float32", out=self.__w00)
        self.__w01 = np.take(self.__cdt, self.__index)
        np.cumsum(self.__w01, axis=1, dtype ="float32", out=self.__w01)
        w00_max = np.amax(self.__w00[:, -1])
        w01_max = np.amax(self.__w01[:, -1])
        """Calculate the weighted error matrix and find the least error"""
        self.__err = evaluate("w00 - w01",local_dict = {'w00': self.__w00, 
                                                        'w01': self.__w01,})
        e1 = w01_max + np.amin(self.__err)
        e2 = w00_max - np.amax(self.__err)
        if e1 < e2:
            err_ind = np.argmin(self.__err)
            err_best = e1
        else:
            err_ind = np.argmax(self.__err)
            err_best = e2
        
        (d1, d2) = np.unravel_index(err_ind, self.__err.shape)

        """Calculate d1 d2 d3 d4"""
        
        if d2 < self.process.train_exam_no - 1: 
            d3 = self.__index[d1, d2] 
            d4 = self.__index[d1, d2] 
        else:
            d3 = self.__index[d1, d2] 
            d4 = d3;
        
        """Calculate c0 and c1"""
        eps = np.float32(1e-3 / self.process.train_exam_no);
        w00_bh = self.__w00[d1, d2]
        w01_bh = self.__w01[d1, d2]
        w10_bh = w00_max - w00_bh
        w11_bh = w01_max - w01_bh
        if err_best < eps:
            err_best = eps
        alpha = 0.5 * np.log((1.0-err_best)/err_best)
        if w00_bh < w01_bh:
            c0 = alpha
        else:
            c0 = -alpha
        if w10_bh < w11_bh:
            c1 = alpha
        else:
            c1 = -alpha
        self.__bout[:] = False
        self.__bout[self.__index[d1,0:d2+1]] = True
        
        val = np.array([err_best,self.pb.rank, d1, d2, d3, d4, c0, c1])
        return val,self.__bout
