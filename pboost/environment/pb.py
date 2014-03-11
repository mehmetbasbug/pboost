import h5py,sys,os,glob,time
import numpy as np
from datetime import datetime
from pboost.utility import parser
from pboost.feature.extract import Extractor
from pboost.boost.process import Process,ProcessMPI
from pboost.report.reporter import Reporter

class PBoost():
    def __init__(self, 
                 conf_dict = None,                 
                 conf_num = None, 
                 conf_path = "./",
                 debugEN = False,
                 logEN = True,
                 ):
        """
        
        Set up the environment for a pboost job
        
        
        Parameters
        ----------
        conf_num : integer
            Configuration number
        
        conf_path : String, optional
            Path to configuration file
            
        logEN : boolean, optional        
            Flag to enable logging for leader job
            
        """
        if conf_dict is not None:
            self.conf_dict = conf_dict
        elif conf_num is not None:    
            self.conf_dict = parser.get_configuration(conf_num = conf_num, 
                                                      conf_path = conf_path)
        else:
            raise Exception('Configuration dictionary or configuration number should be specified.')
        self.conf_num = self.conf_dict['conf_num']
        self.rank = 0
        self.testEN = self.conf_dict['test_file'] != None
        self.xvalEN = self.conf_dict['xval_no'] != 1
        self.xval_no = self.conf_dict['xval_no']
        self.wd = self.conf_dict['working_dir']
        self.example_dim = self.conf_dict['example_dim']
        
        """Filepath to the training dataset"""
        self.train_fp = self.wd + self.conf_dict['train_file']
        
        """Total number of examples in the training dataset"""
        self.total_exam_no = self.get_exam_no(fp = self.train_fp, 
                                              ds_name = 'data')
        
        """Partition of the training dataset between workers"""
        self.train_partition = np.array([0,self.total_exam_no])
        
        """
        Start and end indices for the partition of training dataset this
        worker is responsible for        
        """
        self.train_ind1 = 0
        self.train_ind2 = self.total_exam_no
        
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
            self.test_partition = np.array([0,self.test_exam_no])
            self.test_ind1 = 0
            self.test_ind2 = self.test_exam_no
        
        self.algorithm = self.conf_dict['algorithm']
        self.rounds = self.conf_dict['rounds']
        self.factory_files = self.conf_dict['factory_files']
        self.max_memory = self.conf_dict['max_memory']
        self.show_plots = self.conf_dict['show_plots']
        self.deduplicationEN = self.conf_dict['deduplication']!='n'
        self.deduplication = self.conf_dict['deduplication']
        self.omp_threads = self.conf_dict['omp_threads']
        self.depth = self.conf_dict['tree_depth']
        
        self.isLeader = True
        self.logEN = logEN
        self.debugEN = debugEN
        
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
        self.partition = np.array([0,total_feature_no])
        self.features_ind1 = 0
        self.features_ind2 = total_feature_no
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
        self.partition[1] = span
        
        if self.logEN and self.deduplicationEN:
            dff = self.total_feature_no - self.partition[-1]
            tff = self.total_feature_no
            perc = float(dff) / tff
            msg = "Info : %0.2f percent of features" % (perc,)
            msg = msg + " (%i out of %i) are eliminated." % (dff,tff)
            print datetime.now(),msg
            
        self.total_feature_no = self.partition[-1]
        self.features_ind1 = 0
        self.features_ind2 = self.partition[-1]
        self.features_span = self.partition[-1]
        self.feature_mapping = self.feature_mapping[0:self.features_span]

        
    def set_xval_indices(self,fp,ds_name='indices'):
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
        self.xval_indices = indices
        
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
        exam_no = ds.shape[self.example_dim]
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
            label = np.int16(f["label"][:])
        except KeyError:
            print "Warning : The train label does not exist.Using last column."
            label = np.int16(f['data'][:,-1])
            pass
        f.close()
        return label
    
    def clean(self):
        if self.xvalEN:
            jobs = np.arange(self.xval_no+1)
        else:
            jobs = (0,)
        files_to_del = []
        files_to_del.append(self.model_fp)
        files_to_del.append(self.feature_db)
        for job in jobs:
            files_to_del.append(self.wd + "out_" + str(self.conf_num) + 
                                "_" + str(job) + ".npz")
        for fp in files_to_del:
            try:
                os.remove(fp)
            except OSError:
                pass
    
    def run(self):
        """
        
        Runs the pboost job
        
        """
        tstart = time.time()
        if self.logEN:
            msg = "Info : Pboost is running for configuration number : "
            msg = msg + str(self.conf_num)
            print datetime.now(),msg
        
        if self.isLeader:
            self.clean()
        
        """Extract features"""
        extractor = Extractor(pb=self)
        if self.logEN:
            msg = "Info : Extractor is created."
            print datetime.now(),msg
        
        extractor.extract()
        if self.logEN:
            msg = "Info : Extractor finished running."
            print datetime.now(),msg
        
        if self.logEN:
            msg = "Info : Partition of feature space is  "+str(self.partition)
            print datetime.now(),msg

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
                print datetime.now(),msg
                
            boost_process.run()
            if self.logEN:
                msg = "Info : Boosting computation is finished xval index "
                msg = msg + str(xval_ind)
                print datetime.now(),msg
            boost_process = None
        
        """Report the results"""
        if self.isLeader:
            """Call post processing step"""
            reporter = Reporter(pb = self)
            if self.logEN:
                msg = "Info : Reporter is created."
                print datetime.now(),msg
    
            reporter.run()
            if not self.debugEN:
                self.clean()
            
            if self.logEN:
                msg = "Info : Reporter finished running."
                print datetime.now(),msg
                tfinish = time.time()
                delta = tfinish - tstart
                msg = "Info : Total runtime : %0.2f seconds" % (delta,)
                print datetime.now(),msg
            
    
class PBoostMPI(PBoost):
    def __init__(self, 
                 comm, 
                 conf_num, 
                 conf_path="./",
                 debugEN = False,
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
        PBoost.__init__(self,
                        conf_num = conf_num,
                        conf_path = conf_path,
                        debugEN = debugEN,
                        logEN = logEN,
                        )

        self.comm = comm
        self.rank = comm.Get_rank()
        self.comm_size = comm.Get_size()

        self.model_fp = self.wd + "model_%s_%s.h5" % (conf_num,self.rank)
        
        """Partition of the training dataset between workers"""
        self.train_partition = self.create_partition(limit=self.total_exam_no)
        
        """
        Start and end indices for the partition of training dataset this
        worker is responsible for        
        """
        self.train_ind1 = self.train_partition[self.rank]
        self.train_ind2 = self.train_partition[self.rank+1]
        
        self.test_partition = None
        self.test_ind1 = None
        self.test_ind2 = None
        
        """If test dataset is given, calculate related parameters"""
        if self.testEN:
            self.test_partition = self.create_partition(limit = self.test_exam_no)
            self.test_ind1 = self.test_partition[self.rank]
            self.test_ind2 = self.test_partition[self.rank+1]
        
        self.isLeader = False
        self.logEN = logEN
        self.debugEN = debugEN
        """Choose the worker with zero rank to be the leader"""
        if self.rank == 0:
            self.isLeader = True
            self.logEN = leaderLogEN
        

#         """For consistency sync xval indices"""
#         self.set_xval_indices(fp = self.train_fp, 
#                                ds_name = 'indices')
#                 
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
    
    def set_xval_indices(self,fp,ds_name='indices'):
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
    
    def set_partition(self,
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
    
    def clean(self):
        if self.xvalEN:
            jobs = np.arange(self.xval_no+1)
        else:
            jobs = (0,)
        files_to_del = []
        for rnk in np.arange(self.comm_size):
            model_fp = self.wd + "model_%s_%s.h5" % (self.conf_num,rnk)
            files_to_del.append(model_fp)
        files_to_del.append(self.feature_db)
        for job in jobs:
            files_to_del.append(self.wd + "out_" + str(self.conf_num) + 
                                "_" + str(job) + ".npz")
        for fp in files_to_del:
            try:
                os.remove(fp)
            except OSError:
                pass
    
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
            self.clean()
        
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
            boost_process = ProcessMPI(pb = self,
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
            if not self.debugEN:
                self.clean()
            
            if self.logEN:
                msg = "Info : Reporter finished running."
                print datetime.now(),"Rank",self.rank,msg
                tfinish = time.time()
                delta = tfinish - tstart
                msg = "Info : Total runtime : %0.2f seconds" % (delta,)
                print datetime.now(),"Rank",self.rank,msg
