import os,sqlite3
import numpy as np
from pboost.feature.extract import Feature

class Node():
    def __init__(self):
        self.depth = 1
        self.left = None
        self.right = None
        self.isLeaf = True
        self.isRoot = True
        self.rnk = -1
        self.d1 = -1
        self.d2 = -1
        self.d3 = -1
        self.d4 = -1
        self.d5 = -1
        self.c0 = -1
        self.c1 = -1
        self.v = 0.0
        self.fn_def = None
        self.pred = None
        self.mask = None
    
    def _get_feature_def(self,cursor,rowid):
        s = 'SELECT * FROM features WHERE rowid='+str(rowid)
        cursor.execute(s)
        return list(cursor.fetchone())
    
    def set_fn_def(self,cursor = None):
        if self.d5 is None:
            raise Exception('d5 cannot be None')
        if cursor is None:
            conn = sqlite3.connect(self.pb.feature_db)
            cursor = conn.cursor()        
        fn_def = self._get_feature_def(cursor,self.d5)
        head,tail = os.path.split(fn_def[0])
        root,ext =  os.path.splitext(tail)
        fn_def[0] = './'+root+'.py'
        self.fn_def = fn_def
        
    def to_dict(self,cursor = None):
        if self.fn_def is None:
            self.set_fn_def(cursor)
        d = dict()
        d['fn_def'] = self.fn_def
        d['threshold'] = self.v
        d['isLeaf'] = self.isLeaf
        if not self.isLeaf:
            d['left'] = self.left.to_dict(cursor)
            d['right'] = self.right.to_dict(cursor)
        return d
    
    def from_dict(self,d):
        self.fn_def = d['fn_def']
        self.v = d['threshold']
        self.isLeaf = d['isLeaf']
        if not self.isLeaf:
            node = Node()
            self.left = node.from_dict(d['left'])
            node = Node()
            self.right = node.from_dict(d['right'])

    def insert_child(self,subtree,isLeft = True):
        self.isLeaf = False
        if isLeft:
            self.left = subtree
            self.left.isRoot = False
            if self.pred is not None:
                self.left.mask = self.pred
            self.left.depth = self.depth + 1
        else:
            self.right = subtree
            self.right.isRoot = False
            if self.pred is not None:
                self.right.mask = np.logical_not(self.pred)
            self.right.depth = self.depth + 1
    
    def set_pred(self,data = None,val = None, pred = None, unmasked = False):
        if unmasked:
            self.mask = None
        if data is not None:
            feat = Feature(data = data,
                           feature_def = self.fn_def)
            val = feat.apply(params = self.fn_def[2])
        elif val is not None:
            self.pred = val <= self.v
        elif pred is not None:
            self.pred = pred
        else:
            raise Exception('Data, values or boolean prediction vector should be given as argument')
        if self.mask is not None:
            self.pred = np.logical_and(self.pred,self.mask)

    def predict(self,tree):
        if not self.isRoot:
            assert(self.mask is not None)
        if self.isLeaf:
            tree.pred = np.logical_or(tree.pred,self.pred)
        else:
            self.left.predict(tree)
            self.right.predict(tree)
            
    def unmasked_predict(self,tree,mask=None):
        if self.isLeaf:
            tree.pred = np.logical_or(tree.pred,np.logical_and(self.pred,mask))
        else:
            if self.isRoot:
                left_mask = self.pred
                right_mask = np.logical_not(self.pred)
            else:
                left_mask = np.logical_and(self.pred,mask)
                right_mask = np.logical_and(self.pred,np.logical_not(mask))
            self.left.unmasked_predict(tree,left_mask)
            self.right.unmasked_predict(tree,right_mask)           

    def append_to_list(self,li):
        if self.isLeaf:
            li.append(self)
        else:
            li.append(self)
            self.left.append_to_list(li)
            self.right.append_to_list(li)
            
    def __str__(self):
        if self.isLeaf:
            return '< %s , %s>' % (self.fn_def[1],self.v)
        else:
            return '< %s , %s >\n%sleft %s\n%sright %s' % (self.fn_def[1],self.v,'\t'*self.depth,self.left,'\t'*self.depth,self.right)

    def __unicode__(self):
        return self.__str__()
            
class Tree():
    def __init__(self):
        self.c0 = -1
        self.c1 = -1
        self.root = None
        self.pred = None
    
    def iterative_predict(self):
        '''Assumes boolean prediction vector is set for all nodes'''
        self.iterative_boolean_predict()
        self.pred = self.pred * self.c0 + np.logical_not(self.pred) * self.c1
        return self.pred

    def iterative_boolean_predict(self):
        '''Assumes boolean prediction vector is set for all nodes'''
        try:
            self.root.predict(self)
        except AssertionError:
            self.root.unmasked_predict(self)
            pass
        return self.pred

    def predict(self,data = None, val = None):
        if data is not None and val is not None:
            for node in self.get_all_nodes():
                node.set_pred(data = data,val = val)
        return self.iterative_predict()

    def to_dict(self):
        d = dict()
        d['c0'] = self.c0
        d['c1'] = self.c1
        d['hypothesis'] = self.root.to_dict()
        return d

    def from_dict(self,d):
        self.c0 = d['c0']
        self.c1 = d['c1']
        node = Node()
        self.root = node.from_dict(d['hypothesis'])
        
    def get_all_nodes(self):
        li = list()
        self.root.append_to_list(li)
        return li
    
    def __str__(self):
        return '< %0.2f , %0.2f , \n%s >' % (self.c0,self.c1,self.root)

    def __unicode__(self):
        return self.__str__()
