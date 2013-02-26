import sys,json,os,h5py,sqlite3,time
import numpy as np
import inspect
from pboost.feature.factory import BaseFeatureFactory

class Extractor():
    def __init__(self, pb):
        """
        Create a pre-processing object
        Preparation and parsing the configuration file
        """
        self.pb = pb
        total_feature_no = 0
        if pb.isLeader:
            total_feature_no = self.make_features()
        pb.sync_partition(total_feature_no = total_feature_no)
        pb.sync_xval_indices()
    
    def make_features(self):
        """
        Populate behavior table
        Obtain hypotheses by reading different arguments for each behavior
        """
        conn = sqlite3.connect(self.pb.feature_db)
        c = conn.cursor()
        c.execute("CREATE TABLE features(Cls_Path TEXT, Cls_Name TEXT, Argstr TEXT)")
        conn.commit()
        conn.close()
        f = h5py.File(self.pb.train_fp,'r')
        traindata = f['data']
        total_feature_no = None
        itr = FactoryFileIterator(self.pb.factory_files)
        while(itr.has_next()):
            ff = itr.get_next()
            factory_classes = ff.get_classes()
            for factory_cls in factory_classes:
                factory = factory_cls(data = traindata,
                                      db_path = self.pb.feature_db)
                kwargs = self.pb.conf_dict['arbitrary']
                factory.make(**kwargs)
                total_feature_no = factory.finalize()
        f.close()
        return total_feature_no
        
    def read_features(self): 
        """
        Read behavior table and get the content information
        """
        conn = sqlite3.connect(self.pb.feature_db)
        cursor = conn.cursor()
        s = "SELECT * FROM features WHERE "
        s = s+"rowid BETWEEN %s AND %s"%(self.pb.features_ind1+1,
                                         self.pb.features_ind2)
        cursor.execute(s)
        all_features = cursor.fetchall()
        conn.close()
        return all_features
                
    def create_io_partition(self):
        """
        Create pre-partition list for parallelization
        """
        IO_LIMIT = 128e6
        if self.pb.testEN:
            threshold = int(IO_LIMIT/self.pb.test_exam_no/10)
        else:
            threshold = int(IO_LIMIT/self.pb.total_exam_no/6)
        local_lim = self.pb.features_ind2 - self.pb.features_ind1
        io_partition = np.array([0])
        while True:
            if threshold * io_partition.size >= local_lim:
                io_partition = np.append(io_partition,local_lim)
                break
            else:
                io_partition = np.append(io_partition,
                                         threshold*io_partition.size)
        return io_partition
    
    def extract(self):
        """
        Apply every function to every example and sort
        Write the result to the function output
        Write the sorted column sorting tables
        """

        all_features = self.read_features()
        model_file = h5py.File(self.pb.model_fp,'w')
        io_index = model_file.create_dataset("index",
                                  self.pb.index_matrix.shape,
                                  self.pb.index_matrix.dtype)
        io_train = model_file.create_dataset("train_unsorted",
                                  self.pb.index_matrix.shape,
                                  "float32")  
        if self.pb.testEN:
            io_test = model_file.create_dataset("test_unsorted",
                                      (self.pb.features_span,
                                       self.pb.test_exam_no),
                                       "float32")  

        train_file = h5py.File(self.pb.train_fp,'r')
        traindata = train_file['data']
        if self.pb.testEN:
            test_file = h5py.File(self.pb.test_fp,'r')
            testdata = test_file['data']
        
        current_cls = None
        current_fp =  None
        io_partition = self.create_io_partition()
        io_ind = 0
        for io_ind in np.arange(len(io_partition)-1):
            io_current = io_partition[io_ind]
            io_next = io_partition[io_ind+1]
            feature_chunk = all_features[io_current:io_next]
            index_chunk = np.zeros((io_next-io_current,self.pb.total_exam_no),
                               dtype = self.pb.index_matrix.dtype)
            train_chunk = np.zeros((io_next-io_current,self.pb.total_exam_no),
                               dtype = "float32") 
            if self.pb.testEN:
                test_chunk = np.zeros((io_next-io_current,self.pb.test_exam_no),
                                    dtype = "float32")
            feature_ind = 0
            for feature_def in feature_chunk:
                fp = feature_def[0]
                cls = feature_def[1]
                params = feature_def[2]
                if fp != current_fp or cls != current_cls:
                    current_fp = fp
                    current_cls = cls
                    feature = Feature(feature_def = feature_def,
                                      data = traindata)
                    if self.pb.testEN:
                        test_feature = Feature(feature_def = feature_def,
                                               data = testdata)
                train_vals = feature.apply(params = params)
                train_chunk[feature_ind,:] = train_vals
                if self.pb.testEN:
                    test_vals = test_feature.apply(params = params)
                    test_chunk[feature_ind,:] = test_vals
                ind = np.argsort(train_vals)
                index_chunk[feature_ind,:] = ind
                feature_ind = feature_ind + 1
            io_index[io_current:io_next,:] = index_chunk
            io_train[io_current:io_next,:] = train_chunk
            if self.pb.testEN:
                io_test[io_current:io_next,:] = test_chunk
            self.pb.index_matrix[io_current:io_next,:] = index_chunk
        
        model_file.close()
        train_file.close()
        if self.pb.testEN:
            test_file.close()
        
class Feature():
    
    def __init__(self,
                 data,
                 filepath = None,
                 cls_name = None,
                 feature_def = None,
                 ):
        if feature_def:
            self.filepath = feature_def[0]
            self.cls_name = feature_def[1]
        elif cls_name and filepath:
            self.filepath = filepath
            self.cls_name = cls_name
        else:
            raise Exception("Error : Should give feature definition or classname and filepath")
        head,tail = os.path.split(self.filepath)
        sys.path.append(head)
        root,ext =  os.path.splitext(tail)
        m = __import__(root)
        feat_cls = getattr(m,self.cls_name)
        self.fc =  feat_cls(data)
    
    def apply(self,params):
        args = json.loads(params)
        return self.fc.blueprint(*args)

class FactoryFile():
    def __init__(self,
                 factory_class_names = None,
                 factory_classes = None,
                 fp = None):
        self.factory_class_names = list()
        self.factory_classes = list()
        if fp is not None:
            self._create_with_fp(fp)
        elif factory_class_names and factory_classes:
            self.factory_class_names = factory_class_names
            self.factory_classes = factory_classes
        else:
            raise Exception("Must specify a filepath or factory classes")
        
    def _create_with_fp(self,fp):
        sys.path.append(fp)
        head,tail = os.path.split(fp)
        root,ext = os.path.splitext(tail)
        try:
            m = __import__(root)
            for name, obj in inspect.getmembers(m):
                if inspect.isclass(obj):
                    if issubclass(obj,BaseFeatureFactory):
                        if name.split('.')[-1] != "BaseFeatureFactory":
                            self.factory_class_names.append(name)
                            self.factory_classes.append(getattr(m,name))
        except ImportError:
            msg = 'Error : The file path to the function behaviors does' 
            msg = msg + ' not exist. Please check your configuration.'
            raise Exception(msg)
        
    def get_class_names(self):
        return self.factory_class_names
    
    def get_classes(self):
        return self.factory_classes
    
class FactoryFileIterator():
    
    def __init__(self,factory_files):
        """
        Initialize a FactoryFileIterator object to parse behavior files
        """
        self.factory_files = factory_files
        self.__last = len(factory_files)
        self.__current = 0
    
    def has_next(self):
        return self.__current < self.__last
    
    def get_next(self):
        """
        Return the module and the list of blueprintd classes in this module
        """
        factory_fp = self.factory_files[self.__current]
        if factory_fp == "default":
            m = __import__("pboost.feature.factory",
                           None,
                           locals(),
                           ["DefaultFeatureFactory"],
                           -1)
            factory_classes = [getattr(m,"DefaultFeatureFactory"),]
            factory_class_names = ["DefaultFeatureFactory",]
            ff = FactoryFile(factory_class_names = factory_class_names,
                               factory_classes = factory_classes)
        else:
            ff = FactoryFile(fp = factory_fp)
        self.__current = self.__current + 1
        return ff
        