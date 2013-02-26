import h5py,sys,os,glob
import numpy as np
from datetime import datetime
from pboost.utility import parser
from pboost.feature.extract import Extractor
from pboost.boost.process import Process
from pboost.report.reporter import Reporter

class PBoost():
    def __init__(self, 
                 comm, 
                 conf_num, 
                 conf_path="./", 
                 logEN = False,
                 leaderLogEN=True):
        """
        This class is a simple container to hold necessary objects
        and pass them around
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.conf_dict = parser.get_configuration(conf_num = conf_num, 
                                                  conf_path = conf_path)
        self.conf_num = conf_num
        self.testEN = self.conf_dict['test_file'] != None
        self.xvalEN = self.conf_dict['xval_no'] != 1
        self.xval_no = self.conf_dict['xval_no']
        self.wd = self.conf_dict['working_dir']
        self.train_fp = self.wd + self.conf_dict['train_file']
        self.total_exam_no = self.get_exam_no(fp = self.train_fp, 
                                              ds_name = 'data')
        self.train_partition = self.create_partition(limit = self.total_exam_no)
        self.train_ind1 = self.train_partition[self.rank]
        self.train_ind2 = self.train_partition[self.rank+1]
        self.test_fp = None
        self.test_exam_no = None
        self.test_partition = None
        self.test_ind1 = None
        self.test_ind2 = None
        
        if self.testEN:
            self.test_fp = self.wd + self.conf_dict['test_file']
            self.test_exam_no = self.get_exam_no(fp = self.test_fp, 
                                                 ds_name = 'data')        
            self.test_partition = self.create_partition(limit = self.test_exam_no)
            self.test_ind1 = self.test_partition[self.rank]
            self.test_ind2 = self.test_partition[self.rank+1]
        
        self.algorithm = self.conf_dict['algorithm']
        self.rounds = self.conf_dict['rounds']
        self.factory_files = self.conf_dict['factory_files']
        self.max_memory = self.conf_dict['max_memory']
        self.show_plots = self.conf_dict['show_plots']
        
        self.isLeader = False
        self.logEN = logEN
        if self.rank == 0:
            self.isLeader = True
            self.logEN = leaderLogEN
            
        self.feature_db = self.wd + "features_%s.db" % conf_num
        self.model_fp = self.wd + "model_%s_%s.h5" % (conf_num,self.rank)
        
        self.total_feature_no = None
        self.partition = None
        self.features_ind1 = None
        self.features_ind2 = None
        self.features_span = None
                
        self.index_matrix = None

        self.xval_indices = None
        self.sync_xval_indices()
        """Append working directory to python path"""
        sys.path.append(self.wd) 
        
    def set_partition(self,
                      total_feature_no):
        self.total_feature_no = total_feature_no
        self.partition = self.create_partition(limit = total_feature_no)
        self.features_ind1 = self.partition[self.rank]
        self.features_ind2 = self.partition[self.rank+1]
        self.features_span = self.features_ind2 - self.features_ind1
        if self.total_exam_no < 255:
            dtype = np.dtype('u1')
        elif self.total_exam_no < 65535:
            dtype = np.dtype('u2')
        else:
            dtype = np.dtype('u4')
        self.index_matrix = np.zeros((self.features_span,self.total_exam_no),
                                     dtype = dtype)
    
    def create_partition(self,limit):
        threshold = int(limit/self.comm_size)+1
        partition = np.array([0])  
        while True:
            if threshold * partition.size >= limit:
                partition = np.append(partition,
                                      limit)
                break
            else:
                partition = np.append(partition,
                                      threshold*partition.size)
        return partition
    
    def sync_xval_indices(self):
        indices = None
        if self.isLeader:
            indices = np.repeat(np.arange(1, (self.xval_no + 1)), 
                                self.total_exam_no / self.xval_no)
            np.random.shuffle(indices)
        indices = self.comm.bcast(indices,root = 0)
        self.xval_indices = indices
    
    def sync_partition(self,
                       total_feature_no):
        total_feature_no = self.comm.bcast(total_feature_no,root = 0)
        self.set_partition(total_feature_no)
        
    def get_exam_no(self,
                    fp,
                    ds_name="data"):
        f = h5py.File(fp,'r')
        ds = f[ds_name]
        exam_no = ds.shape[0]
        f.close()
        return exam_no
    
    def get_label(self,
                  source = 'train'):
        if source == 'train':
            f = h5py.File(self.train_fp,'r')
        elif source == 'test':
            f = h5py.File(self.test_fp,'r')
        else:
            raise Exception('Error : Unknown source. Should give train'+ 
                            ' or test as source')
        try:
            label = f["label"]
        except KeyError:
            print "Warning : The train label does not exist.Using last column."
            label = f['data'][:,-1]
            pass
        f.close()
        return label
    
    def run(self):
        if self.logEN:
            msg = "Info : Pboost is running for configuration number : "
            msg = msg + str(self.conf_num)
            print datetime.now(),"Rank",self.rank,msg
        
        if self.isLeader:
            """
            This part is only to make sure that all existing h5 files are deleted
            before run. Should be removed after debugging.
            """
            for filename in glob.glob(self.wd + '*.npz'):
                os.remove(filename)
    
            for filename in glob.glob(self.wd + '*.npy'):
                os.remove(filename)
    
            for filename in glob.glob(self.wd + '*.h5'):
                os.remove(filename)
                
            for filename in glob.glob(self.wd + '*.db'):
                os.remove(filename)
        
        """Call pre processing step"""
        extractor = Extractor(pb=self)
        if self.logEN:
            msg = "Info : Pre-process model is created."
            print datetime.now(),"Rank",self.rank,msg
        
        extractor.extract()
        if self.logEN:
            msg = "Info : Pre-process computation is finished."
            print datetime.now(),"Rank",self.rank,msg
        
        """Wait for all threads to finish their pre processing jobs"""
        self.comm.Barrier()
    
        if self.logEN:
            msg = "Info : Partition of hypotheses space is  "+str(self.partition)
            print datetime.now(),"Rank",self.rank,msg
        
        if self.xvalEN:
            jobs = np.arange(self.xval_no+1)
        else:
            jobs = (0,)
    
        for xval_ind in jobs:        
            boost_process = Process(pb = self,
                                    xval_ind = xval_ind
                                    )
            if self.logEN:
                msg = "Info : Boosting process is created for xval index "
                msg = msg + str(xval_ind)
                print datetime.now(),"Rank",self.rank,msg
                
            boost_process.compute()
            if self.logEN:
                msg = "Info : Boosting computation is finished xval index "
                msg = msg + str(xval_ind)
                print datetime.now(),"Rank",self.rank,msg
            boost_process = None
            
        if self.isLeader:
            """Call post processing step"""
            reporter = Reporter(pb = self)
            if self.logEN:
                msg = "Info : Post process is created."
                print datetime.now(),"Rank",self.rank,msg
    
            reporter.run()
            if self.logEN:
                msg = "Info : Post process is finished."
                print datetime.now(),"Rank",self.rank,msg
