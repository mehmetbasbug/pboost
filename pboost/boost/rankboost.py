import h5py,os,sqlite3
import numpy as np
from pboost.boost.generic import Boosting, WeakLearner
from scipy import weave
from bitarray import bitarray
from mpi4py import MPI
from pboost.boost.decision_tree import Node,Tree

class RankBoost(Boosting):
    def __init__(self, boosting_p):
        """
        
        RankBoost Implementation
        
        Parameters
        ----------

        boosting_p : pboost.boost.process.Process object
            Boosting environment in which this algoritm runs
            
        """
        self.process = boosting_p
        self.pb = boosting_p.pb
        
        self.hypotheses = list()
        self.val_predictions = None
        self.train_predictions = None
        self.test_predictions = None

        """Create empty prediction matrices"""
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

        """Convert binary labesl to +1/-1 form"""
        self.tl = np.int16(np.copy(self.process.label))
        self.tl[self.process.label == 0] = -1
        self.f = np.zeros(self.pb.total_exam_no,dtype = 'float32')
        
        if self.pb.xvalEN and not self.process.isXvalMain:
            self.pos = np.logical_and(self.process.label,
                                      np.logical_not(self.process.val_indices))
            self.neg = np.logical_and(np.logical_not(self.process.label),
                                      np.logical_not(self.process.val_indices))
        else:
            self.pos = self.process.label
            self.neg = np.logical_not(self.process.label)

    def run(self, dt, r, tree):
        """
        
        Run a single round of boosting
        
        
        Parameters
        ----------
        
        dt : numpy float array
            Probability distribution over examples
        
        r : integer
            Round number
        
        rnk : integer
            Rank of the core containing best hyp in its space
            
        d1 : integer
            Index of the best hypothesis
        
        d2 : integer
            The relative index of the example where threshold is placed
            
        d3 : integer
            The actual index of the example where threshold is placed
            
        d4 : integer
            The actual index of the next example in threshold calculation
        c0 : float
            Prediction for values lower than threshold
        
        c1 : float
            Prediction for values larger than threshold
        
        bout : numpy boolean array
            Marker for each example best hyp making a mistake
            
        Returns:
            Updated distribution over examples

        """

                
        """Create a dictionary for the best hypothesis"""
        
        self.hypotheses.append(tree)
        c0 = tree.c0
        c1 = tree.c1
        bout = tree.pred
        
        """Update training and validation predictions"""
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
        
        """Update distribution over the examples"""
        nbout = np.logical_not(bout)
        if self.pb.xvalEN and not self.process.isXvalMain:
            bout = np.logical_and(bout,
                                  np.logical_not(self.process.val_indices))
            nbout = np.logical_and(nbout,
                                   np.logical_not(self.process.val_indices))
        
        zp = 2 * np.sum(np.exp(-self.f[self.pos]))
        zn = 2 * np.sum(np.exp(self.f[self.neg]))
        dt[self.pos] = np.exp(-self.f[self.pos]) / zp
        dt[self.neg] = np.exp(self.f[self.neg]) / zn
#         dt = dt / np.sum(dt)
        self.f[bout] = self.f[bout] + 0.5*np.float32(c0)
        self.f[nbout] = self.f[nbout] + 0.5*np.float32(c1)
        return dt
    
    def finalize(self):
        """
        Writes hypotheses and predictions into a file
        """
        
        """Creates a dictionary of rank,rounds pairs"""
        inverse = dict()
        rnk_list = list()
        for r in np.arange(self.pb.rounds):
            h = self.hypotheses[r]
            for node in h.get_all_nodes():
                rnk = node.rnk
                try:
                    inverse[rnk].append(node)
                except KeyError:
                    inverse[rnk] = [node,]
                    rnk_list.append(rnk)
                    pass
        
        """
        For each rank read the hypotheses space to update threshold and test
        predictions
        """
        for rnk in rnk_list:
            model_fp = self.pb.wd + "model_%s_%s.h5" % (self.pb.conf_num,rnk)
            try:
                mf = h5py.File(model_fp,'r')
            except Exception as e:
                print model_fp
                print e
            unsorted_ds = mf["train_unsorted"]
            for node  in inverse[rnk]:
                d1 = node.d1
                d3 = node.d3
                d4 = node.d4
                vec = unsorted_ds[d1, :]
                node.v = (vec[d3]+vec[d4])/2.0
                if self.pb.testEN:
                    tVals = mf["test_unsorted"][d1,self.pb.test_ind1:self.pb.test_ind2]
                    node.set_pred(val = tVals, unmasked = True)
            mf.close()
        
        if self.pb.testEN:
            for r in np.arange(self.pb.rounds):
                h = self.hypotheses[r]
                h.pred = np.zeros(self.pb.test_ind2 - self.pb.test_ind1)
                self.test_predictions[:,r] = h.predict()
            np.cumsum(self.test_predictions, axis=1, out=self.test_predictions)
        
    def get_hypotheses(self):
        """
        Returns the hypotheses
        """
        return self.hypotheses
    
    def get_val_predictions(self):
        """
        Returns validation predictions
        """
        return self.val_predictions
    
    def get_train_predictions(self):
        """
        Returns training predictions
        """
        return self.train_predictions
    
    def get_test_predictions(self):
        """
        Returns testing predictions
        """
        return self.test_predictions
            
class RankBoostWL(WeakLearner):
    
    def __init__(self, boosting_p):
        """
        
        Decision Stump Implementation compatible with RankBoost
        
        Parameters
        ----------

        boosting_p : pboost.boost.process.Process object
            Boosting environment in which this algoritm runs
            
        """
        self.process = boosting_p
        self.pb = boosting_p.pb

        """Allocate memory for intermediate data matrices"""
        self.__index = self.pb.index_matrix
        self.__err = np.zeros(shape = self.__index.shape,dtype ="float32")
        self.__bout = np.zeros(self.__index.shape[1],dtype="bool")
        self.__dt = np.zeros(shape = self.process.label.shape,dtype ="float32")
        self.__not_label = np.logical_not(self.process.label)
        self.__label = np.logical_not(self.__not_label)
    
    def run(self,dt):
        tree = Tree()
        tree.pred = np.zeros(self.pb.total_exam_no,dtype='bool')
        root = Node()
        tree.root = self.construct_tree(root,dt)
        tree.iterative_boolean_predict()
        w00 = 0.0
        w01 = 0.0
        w10 = 0.0
        w11 = 0.0
        for k,w in enumerate(dt):
            if tree.pred[k]:
                if self.__label[k]:
                    w01 = w01 + w
                else:
                    w00 = w00 + w
            else:
                if self.__label[k]:
                    w11 = w11 + w
                else:
                    w10 = w10 + w
        err_best = min(w00,w01)+min(w10,w11)
        eps = np.float32(1e-3 / self.process.train_exam_no);
        if err_best < eps:
            err_best = eps
        alpha = 0.5 * np.log((1.0-err_best)/err_best)
        if w00 < w01:
            c0 = alpha
        else:
            c0 = -alpha
        if w10 < w11:
            c1 = alpha
        else:
            c1 = -alpha
        tree.c0 = c0
        tree.c1 = c1
        return tree
        
    def construct_tree(self,node,dt):
        if node.depth == self.pb.depth:
            return self.update_node(node,dt)
        else:
            node = self.update_node(node,dt)
            left_node = Node()
            node.insert_child(left_node,isLeft=True)
            right_node = Node()
            node.insert_child(right_node,isLeft=False)
            self.construct_tree(left_node,dt)
            self.construct_tree(right_node,dt)
            return node
        
    def update_node(self,node,distribution):
        dt = np.copy(distribution)
        if node.mask is not None:
            dt[node.mask] = 0.0
        if self.pb.omp_threads==1:
            val,bout = self.single_thread(dt)
        else:
            val,bout = self.multi_thread(dt)
        (rnk, d1, d2, d3, d4, d5, c0, c1) = val[1:9]
        rnk = int(rnk)
        node.rnk = rnk
        node.d1 = d1
        node.d2 = d2
        node.d3 = d3
        node.d4 = d4
        node.d5 = d5
        node.c0 = c0
        node.c1 = c1
        node.set_pred(pred = bout)
        return node

    
    def single_thread(self,dt):
#         dt = np.copy(dt)
        label = self.__label
        index = self.__index
        F = index.shape[0]/4*4
        N = index.shape[1]
        code = """
               #include <cmath>
               float e = 0.0;
               float err_best = 1000.0;
               int d1 = -1;
               int d2 = -1;
               int d3 = -1;
               int d4 = -1;
               float wlrn = 0.0;
               float wlrp = 0.0;
               float wln = 0.0;
               float wlp = 0.0;
               float wrn = 0.0;
               float wrp = 0.0;
               
               float wlp_best = 0.0;
               float wln_best = 0.0;
               float wrn_best = 0.0;
               float wrp_best = 0.0;
               float dtp[N];
               bool labelp[N]; 
               for (int j=0; j<N; ++j){
                   dtp[j] = dt(j);
                   labelp[j] = label(j);
                   if (label(j) == 1){
                       wlrp = wlrp + dt(j);
                   }
                   else{
                       wlrn = wlrn + dt(j);
                   }
               }
               int ii = 0;
               for (int k=0; k<F; ++k ){
                   wln = 0.0;
                   wlp = 0.0;
                   wrp = wlrp;
                   wrn = wlrn;
                   for (int j=0; j<N; ++j ){
                       ii = index(k,j);
                       if (labelp[ii] == 1){
                            wrp = wrp - dtp[ii];
                            wlp = wlp + dtp[ii];
                       }
                       else{
                            wrn = wrn - dtp[ii];
                            wln = wln + dtp[ii];
                       }
                       e = sqrt(wlp*wln)+sqrt(wrp*wrn);
                       if (e < err_best){
                           err_best = e;
                           d1 = k;
                           d2 = j;
                           wlp_best = wlp;
                           wln_best = wln;
                           wrp_best = wrp;
                           wrn_best = wrn;
                       }
                   }
               }
               d3 = index(d1,d2);
               d4 = d3;
               
               float eps = 0.001 / N;
               if (err_best < eps){
                   err_best = eps;
               }
               float c0 = 0.0;
               float c1 = 0.0;
               float err_best_n = 1.0 - err_best;
               float alpha = 0.5 * std::log((err_best_n) / err_best);
               if (wln_best < wlp_best){
                    c0 = alpha;
               }
               else{
                    c0 = -alpha;
               }
                
               if (wrn_best < wrp_best){
                    c1 = alpha;
               }
               else{
                    c1 = -alpha;
               }
               
               py::tuple results(7);
               results[0] = err_best;
               results[1] = d1;
               results[2] = d2;
               results[3] = d3;
               results[4] = d4;
               results[5] = c0;
               results[6] = c1;
    
               return_val = results;
               """
        
        rtrn = weave.inline(code,
                            ['dt','label','index', 'N', 'F'],
                            type_converters = weave.converters.blitz,
                            compiler = 'gcc',
                            extra_compile_args = ['-O3','-ffast-math'])
        (err_best,d1,d2,d3,d4,c0,c1) = rtrn
        d1 = int(d1)
        d2 = int(d2)
        d3 = int(d3)
        d4 = int(d4)
        bout = np.zeros(N,dtype="bool")
        bout[index[d1,0:d2+1]] = True
        d5 = self.pb.feature_mapping[d1]
        val = np.array([err_best,self.pb.rank, d1, d2, d3, d4, d5, c0, c1])
        return val,bout
    
class RankBoostWLMPI(RankBoostWL):
        
    def update_node(self,node,distribution):
        dt = np.copy(distribution)
        if node.mask is not None:
            dt[node.mask] = 0.0
        if self.pb.omp_threads==1:
            val,bout = self.single_thread(dt)
        else:
            val,bout = self.multi_thread(dt)
        new = self.pb.comm.allreduce(val[0],None, MPI.MINLOC)
        val = self.pb.comm.bcast(val, root=new[1])
        bout_ba = bitarray(list(bout))
        bout_c = bout_ba.tobytes()
        bout_c = self.pb.comm.bcast(bout_c, root=new[1])
        bout_ba = bitarray()
        bout_ba.frombytes(bytes(bout_c))
        bout = np.array(bout_ba.tolist()[0:self.pb.total_exam_no])
        (rnk, d1, d2, d3, d4, d5, c0, c1) = val[1:9]
        rnk = int(rnk)
        node.rnk = rnk
        node.d1 = d1
        node.d2 = d2
        node.d3 = d3
        node.d4 = d4
        node.d5 = d5
        node.c0 = c0
        node.c1 = c1
        node.set_pred(pred = bout)
        return node
