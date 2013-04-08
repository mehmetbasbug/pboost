import sys,json,os,h5py,sqlite3
import numpy as np
import inspect
import pybloomfilter
from pboost.feature.factory import BaseFeatureFactory
from bitarray import bitarray

class Extractor():
    def __init__(self, pb):
        """

        Feature extractor class
        
        Parameters
        ----------

        pb : pboost.environment.pb object
            Contains data related to whole program
            
        """
        self.pb = pb
        total_feature_no = 0
        if pb.isLeader:
            total_feature_no = self.produce_features()
        pb.sync_partition(total_feature_no = total_feature_no)
        pb.sync_xval_indices(fp = pb.train_fp, 
                             ds_name = 'indices')
    
    def produce_features(self):
        """
        
        Produce features for given feature factories and populate the 
        database holding description of each particular feature
        
        Returns total feature number after populating db

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
                factory.produce(**kwargs)
                total_feature_no = factory.finalize()
        f.close()
        return total_feature_no
        
    def read_features(self): 
        """
        
        Read the related feature descriptions from the database 
        
        Returns list of feature descriptions

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
        
        Creates IO partition of feature space to write data as chunks
        
        Returns a numpy array of border indices
        
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
        
        Applies features to training and testing data and writes the results 
        to the specified model file. Updates the index_matrix of pboost. 

        """

        all_features = self.read_features()
        if self.pb.deduplicationEN:
            if self.pb.deduplication == 'BloomFilter':
                el = BloomFilter(capacity = len(all_features),
                                 error_rate = 0.01)
            elif self.pb.deduplication == 'InversionCounter':
                el = InversionCounter(threshold = self.pb.total_exam_no/10,
                                validated_set = self.pb.index_matrix)
            elif self.pb.deduplication == 'MapFilter':
                el = MapFilter(arraylength = self.pb.total_exam_no,
                               threshold = 0.1)
            else:
                raise Exception("Error : Unknown feature deduplication method.")
                
        """Create model file and necessary datasets"""
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

        """Read training and testing data"""
        train_file = h5py.File(self.pb.train_fp,'r')
        traindata = train_file['data']
        if self.pb.testEN:
            test_file = h5py.File(self.pb.test_fp,'r')
            testdata = test_file['data']
        
        current_cls = None
        current_fp =  None
        io_partition = self.create_io_partition()
        io_ind = 0
        last_wi = 0
        """Process a single io chunk at a time"""
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
            for f_idx,feature_def in enumerate(feature_chunk):
                fp = feature_def[0]
                cls = feature_def[1]
                params = feature_def[2]
                """Create a new Feature object only when necessary"""
                if fp != current_fp or cls != current_cls:
                    current_fp = fp
                    current_cls = cls
                    feature = Feature(feature_def = feature_def,
                                      data = traindata)
                    if self.pb.testEN:
                        test_feature = Feature(feature_def = feature_def,
                                               data = testdata)
                train_vals = feature.apply(params = params)
                ind = np.argsort(train_vals)
                is_valid = True
                if self.pb.deduplicationEN:
                    is_valid = el.validate(new = ind)
                        
                if is_valid:
                    index_chunk[feature_ind,:] = ind
                    self.pb.index_matrix[last_wi+feature_ind,:] = ind
                    train_chunk[feature_ind,:] = train_vals
                    if self.pb.testEN:
                        test_vals = test_feature.apply(params = params)
                        test_chunk[feature_ind,:] = test_vals
                    self.pb.feature_mapping[feature_ind] = self.pb.features_ind1 + f_idx + io_current+1
                    feature_ind = feature_ind + 1
                
            """Update related data structures as chunks"""
            next_wi = last_wi + feature_ind
            io_index[last_wi:next_wi,:] = index_chunk[0:feature_ind]
            io_train[last_wi:next_wi,:] = train_chunk[0:feature_ind]
            if self.pb.testEN:
                io_test[last_wi:next_wi,:] = test_chunk[0:feature_ind]
#             self.pb.index_matrix[last_wi:next_wi,:] = index_chunk[0:feature_ind]
            last_wi = next_wi
               
        if self.pb.deduplicationEN:
            el.finalize()
            self.pb.index_matrix = self.pb.index_matrix[0:next_wi,:]
            self.pb.adjust_partition(span = next_wi)

        model_file.close()
        train_file.close()
        if self.pb.testEN:
            test_file.close()

class InversionCounter():
    def __init__(self,
                 threshold,
                 validated_set
                 ):
        self.threshold = threshold
        self.validated_set = validated_set
        self.counter = 0

    def merge_and_count(self,a, b):
        c = []
        count = 0
        i, j = 0, 0
        while i < len(a) and j < len(b):
            c.append(min(b[j], a[i]))
            if b[j] < a[i]:
                count += len(a) - i
                j+=1
            else:
                i+=1
        c += a[i:] + b[j:]
        return count, c

    def sort_and_count(self,L):
        if len(L) == 1: return 0, L
        n = len(L) // 2 
        a, b = L[:n], L[n:]
        ra, a = self.sort_and_count(a)
        rb, b = self.sort_and_count(b)
        r, L = self.merge_and_count(a, b)
        return ra+rb+r, L

    def get_permutation(self,L1, L2):
        permutation = map(dict((v, i) for i, v in enumerate(L1)).get, L2)
        return permutation
    
    def validate(self,new):
        if self.counter == 0:
            self.counter = 1
            return True
        else:
            for k in np.arange(self.counter):
                x = self.validated_set[k,:]
                perm = self.get_permutation(x,new)
                inv = self.sort_and_count(perm)[0]
                if inv < self.threshold:
                    return False
            self.counter = self.counter + 1
            return True
    
    def finalize(self):
        return True
    
class MapFilter():
    def __init__(self,
                 arraylength,
                 threshold):
        self.d = set()
        self.ba = bitarray(arraylength)
        self.threshold = arraylength*threshold
        
    def validate(self,new):
        is_valid = False
        self.ba.setall(0)
        count = 0
        toadd = set()
        for j in new:
            self.ba[j] = 1
            s = self.ba.tobytes()
            if s in self.d:
                count = count + 1
            else:
                toadd.add(s)
        if count < self.threshold:
            self.d.update(toadd)
            is_valid = True    
        return is_valid
    
    def finalize(self):
        return True
    

class BloomFilter():
    def __init__(self,
                 capacity,
                 error_rate
                 ):
        self.bf = pybloomfilter.BloomFilter(capacity,error_rate,None)
   
    def validate(self,new):
        ind_str = np.ndarray.tostring(new)
        return not self.bf.add(ind_str)
    
    def finalize(self):
        self.bf.clear_all()
        return True
        
class Feature():
    def __init__(self,
                 data,
                 filepath = None,
                 cls_name = None,
                 feature_def = None,
                 ):
        """

        Feature class
        
        Parameters
        ----------

        data : HDF5 handler
            train or test data handler
        filepath : String, optional
            filepath to a file containing FeatureFactory class
        cls_name : String, optional
            FeatureFactory class name
        feature_def : List , optional
            List of filepath,cls_name and parameters
            
        """
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
        """

        Returns the result of feature with given parameters
        
        Parameters
        ----------

        params : List
            List of arguments
            
        """
        args = json.loads(params)
        return self.fc.blueprint(*args)

class FactoryFile():
    def __init__(self,
                 factory_class_names = None,
                 factory_classes = None,
                 fp = None):
        """

        Factory File class
        
        Parameters
        ----------

        factory_class_names : List of Strings, optional
            names of classes in the file
        factory_classes : List of pboost.feature.factory.BaseFactoryFile 
                          objects, optional
            classes in the file
        fp : String, optional
            Filepath to the factory file
            
        """
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
        """

        Creates Factory File object using filepath
        
        Parameters
        ----------

        fp : String
            Filepath to the factory file
        
        """
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

        Creates Factory File Iterator object
        
        Parameters
        ----------

        factory_files : List of Strings
            Filepaths to feature factory files
        
        """
        self.factory_files = factory_files
        self.__last = len(factory_files)
        self.__current = 0
    
    def has_next(self):
        return self.__current < self.__last
    
    def get_next(self):
        """
        
        Creates the next FactoryFile object in the list
        Returns the object
    
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
        