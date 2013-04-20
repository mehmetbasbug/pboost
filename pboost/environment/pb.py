import h5py,sys,os,glob,time
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
        
        Set up the environment for a pboost job
        
        
        Parameters
        ----------
        comm : mpi4py.MPI.COMM_WORLD object
            Intracommunicator for MPI  
            
        conf_num : integer
            Configuration number
        
        conf_path : String, optional
            Path to configuration file
            
        logEN : boolean, optional        
            Flag to enable logging for non-leader jobs
        
        leaderLogEN : boolean, optional
            Flag to enable logging for the leader
            
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

        """Filepath to the training dataset"""
        self.train_fp = self.wd + self.conf_dict['train_file']
        
        """Total number of examples in the training dataset"""
        self.total_exam_no = self.get_exam_no(fp = self.train_fp, 
                                              ds_name = 'data')
        
        """Partition of the training dataset between workers"""
        self.train_partition = self.create_partition(limit=self.total_exam_no)
        
        """
        Start and end indices for the partition of training dataset this
        worker is responsible for        
        """
        self.train_ind1 = self.train_partition[self.rank]
        self.train_ind2 = self.train_partition[self.rank+1]
        
        self.test_fp = None
        self.test_exam_no = None
        self.test_partition = None
        self.test_ind1 = None
        self.test_ind2 = None
        
        """If test dataset is given, calculate related parameters"""
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
        self.deduplicationEN = self.conf_dict['deduplication']!='n'
        self.deduplication = self.conf_dict['deduplication']
        self.omp_threads = self.conf_dict['omp_threads']
        
        self.isLeader = False
        self.logEN = logEN
        
        """Choose the worker with zero rank to be the leader"""
        if self.rank == 0:
            self.isLeader = True
            self.logEN = leaderLogEN
        
        """Filepath to the database holding feature descriptions """
        self.feature_db = self.wd + "features_%s.db" % conf_num
        
        """Filepath to the HDF5 file holding intermediate data """
        self.model_fp = self.wd + "model_%s_%s.h5" % (conf_num,self.rank)
        
        self.total_feature_no = None
        self.partition = None
        self.features_ind1 = None
        self.features_ind2 = None
        self.features_span = None
        self.feature_mapping = None
                
        self.index_matrix = None
        self.xval_indices = None
        
        """For consistency sync xval indices"""
        self.sync_xval_indices(fp = self.train_fp, 
                               ds_name = 'indices')
        
        """Append working directory to python path"""
        sys.path.append(self.wd) 
        
    def set_partition(self,
                      total_feature_no):
        """
        
        Partitions the feature space between workers 
        Sets start and end indices for the subspace of this worker
        Initializes the index_matrix according to the subspace's size
        
        Parameters
        ----------
        total_feature_no : integer
            Number of features to partition
            
        """
        self.total_feature_no = total_feature_no
        self.partition = self.create_partition(limit = total_feature_no)
        self.features_ind1 = self.partition[self.rank]
        self.features_ind2 = self.partition[self.rank+1]
        self.features_span = self.features_ind2 - self.features_ind1
        self.feature_mapping = np.zeros(self.features_span)
        
        """Choose the shortest length datatype possible"""
        if self.total_exam_no < 255:
            dtype = np.dtype('u1')
        elif self.total_exam_no < 65535:
            dtype = np.dtype('u2')
        else:
            dtype = np.dtype('u4')
        self.index_matrix = np.zeros((self.features_span,self.total_exam_no),
                                     dtype = dtype)
        
    def adjust_partition(self,
                         span):
        if self.isLeader:
            self.partition[1] = span
            for slv in np.arange(1,self.comm_size):
                slv_span = self.comm.recv(source=slv,tag=101) 
                self.partition[slv+1] = self.partition[slv] + slv_span
        else:
            self.comm.send(span,dest = 0,tag=101)
        
        self.partition = self.comm.bcast(self.partition,root = 0)
        
        if self.logEN and self.deduplicationEN:
                dff = self.total_feature_no - self.partition[-1]
                tff = self.total_feature_no
                perc = float(dff) / tff
                msg = "Info : %0.2f percent of features" % (perc,)
                msg = msg + " (%i out of %i) are eliminated." % (dff,tff)
                print datetime.now(),"Rank",self.rank,msg
            
        self.total_feature_no = self.partition[-1]
        self.features_ind1 = self.partition[self.rank]
        self.features_ind2 = self.partition[self.rank+1]
        self.features_span = self.features_ind2 - self.features_ind1
        self.feature_mapping = self.feature_mapping[0:self.features_span]
        
    def create_partition(self,limit):
        """
        
        Partitions a space between workers. 
        
        Returns a numpy array of border indices
        
        Parameters
        ----------
        limit : integer
            Number of items to partition
        
        """
        threshold = int(np.ceil(float(limit)/self.comm_size))
        if threshold < 1:
            raise Exception("Error : Too many cores for the given dataset size.")
#        threshold = int(limit/self.comm_size)+1
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
    
    def sync_xval_indices(self,fp,ds_name='indices'):
        """
        
        For leader, creates an array of random integers sampled 
        from 1 to xval_no+1 as xval_indices and send this array
        to other workers
        
        For others, receives the xval_indices from the leader
                
        """
        indices = None
        if self.isLeader:
            f = h5py.File(fp,'r')
            try:
                indices = f[ds_name][:]
                if np.amax(indices) != self.xval_no:
                    s = "Warning : Given xval indices are not compatible with "
                    s = s + "xval no. Randomly creating indices."
                    print s
                    indices = np.repeat(np.arange(1, (self.xval_no + 1)), 
                                        self.total_exam_no / self.xval_no)
                    np.random.shuffle(indices)
            except KeyError:
                s = "Warning : Xval indices are not found in the datafile."
                s = s + "Randomly creating indices."
                print s
                indices = np.repeat(np.arange(1, (self.xval_no + 1)), 
                                    self.total_exam_no / self.xval_no)
                np.random.shuffle(indices)
                pass
            f.close()
        indices = self.comm.bcast(indices,root = 0)
        self.xval_indices = indices
    
    def sync_partition(self,
                       total_feature_no):
        """
        
        For leader, send total_feature_no to other workers
        For others, receives total_feature_no from the leader
        Calls set_partition method
        
        Parameters
        ----------
        total_feature_no : integer
            Number of features to partition
        
        """
        total_feature_no = self.comm.bcast(total_feature_no,root = 0)
        self.set_partition(total_feature_no)
        
    def get_exam_no(self,
                    fp,
                    ds_name="data"):
        """
        
        Returns the example number from a dataset
        
        Parameters
        ----------
        fp : String
            Filepath to the data file
            
        ds_name : String, optional
            Name of the dataset 
        
        """
        f = h5py.File(fp,'r')
        ds = f[ds_name]
        exam_no = ds.shape[0]
        f.close()
        return exam_no
    
    def get_label(self,
                  source = 'train'):
        """
        
        Returns the label from a dataset
        
        Parameters
        ----------
        source : String, optional
            The source of the data, "train" or "test"
            
        """
        
        if source == 'train':
            f = h5py.File(self.train_fp,'r')
        elif source == 'test':
            f = h5py.File(self.test_fp,'r')
        else:
            raise Exception('Error : Unknown source. Should give train'+ 
                            ' or test as source')
        try:
            label = f["label"][:]
        except KeyError:
            print "Warning : The train label does not exist.Using last column."
            label = f['data'][:,-1]
            pass
        f.close()
        return label
    
    def run(self):
        """
        
        Runs the pboost job
        
        """
        tstart = time.time()
        if self.logEN:
            msg = "Info : Pboost is running for configuration number : "
            msg = msg + str(self.conf_num)
            print datetime.now(),"Rank",self.rank,msg
        
        if self.isLeader:
            """
            This part is only to make sure that all existing intermediate 
            files are deleted before run. Should be removed after debugging.
            """
            for filename in glob.glob(self.wd + '*.npz'):
                os.remove(filename)
    
            for filename in glob.glob(self.wd + '*.npy'):
                os.remove(filename)
    
            for filename in glob.glob(self.wd + '*.h5'):
                os.remove(filename)
                
            for filename in glob.glob(self.wd + '*.db'):
                os.remove(filename)
        
        """Extract features"""
        extractor = Extractor(pb=self)
        if self.logEN:
            msg = "Info : Extractor is created."
            print datetime.now(),"Rank",self.rank,msg
        
        extractor.extract()
        if self.logEN:
            msg = "Info : Extractor finished running."
            print datetime.now(),"Rank",self.rank,msg
        
        """Wait for all workers to finish their extraction job"""
        self.comm.Barrier()
    
        if self.logEN:
            msg = "Info : Partition of feature space is  "+str(self.partition)
            print datetime.now(),"Rank",self.rank,msg
        
        if self.xvalEN:
            jobs = np.arange(self.xval_no+1)
        else:
            jobs = (0,)
    
        """Run boosting for each xval index"""
        for xval_ind in jobs:        
            boost_process = Process(pb = self,
                                    xval_ind = xval_ind
                                    )
            if self.logEN:
                msg = "Info : Boosting process is created for xval index "
                msg = msg + str(xval_ind)
                print datetime.now(),"Rank",self.rank,msg
                
            boost_process.run()
            if self.logEN:
                msg = "Info : Boosting computation is finished xval index "
                msg = msg + str(xval_ind)
                print datetime.now(),"Rank",self.rank,msg
            boost_process = None
        
        """Report the results"""
        if self.isLeader:
            """Call post processing step"""
            reporter = Reporter(pb = self)
            if self.logEN:
                msg = "Info : Reporter is created."
                print datetime.now(),"Rank",self.rank,msg
    
            reporter.run()
            if self.logEN:
                msg = "Info : Reporter finished running."
                print datetime.now(),"Rank",self.rank,msg
                tfinish = time.time()
                delta = tfinish - tstart
                msg = "Info : Total runtime : %0.2f seconds" % (delta,)
                print datetime.now(),"Rank",self.rank,msg
