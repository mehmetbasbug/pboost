import os, shutil, glob, inspect,sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pboost.report import pyroc
from pboost.report import final_classifier_script
from pboost.feature import factory

class Reporter():
    def __init__(self, pb):
        """
        
        Set up the environment for post-process
        
        pb
            PBoost object containing global variables for the whole program
        meta_dict
            A dictionary containing information from pre-process
            
        """
        self.pb = pb
        self.train_label = pb.get_label(source = 'train')
        self.test_label = None
        
        """default values None. Over-ridden if necessary"""
        self.train_predictions = None
        self.test_predictions = None
        self.hypotheses = None
        self.validation_predictions = None
        self.out_fp = None
        
        self._create_dir()
        if self.pb.testEN:
            self.test_label = pb.get_label(source = 'test')
            self._read_train_and_test()
        else:
            self._read_train()

        if self.pb.xvalEN:
            self.validation_predictions = np.zeros(
                                          [self.pb.total_exam_no,self.pb.rounds],
                                          dtype="float32"
                                          )
            self._combine_val()
        
    def _read_train(self):
        """
        These two methods read the hypotheses and predictions 
        stored on disk by the boosting module 
        """
        outfilename = str(self.pb.wd + 
                          "out_"+str(self.pb.conf_num)+"_0.npz")
        outfile = np.load(outfilename)
        self.train_predictions = outfile['train_predictions']
        self.hypotheses = outfile['hypotheses']
    
    def _read_train_and_test(self):
        outfilename = str(self.pb.wd + 
                          "out_"+str(self.pb.conf_num)+"_0.npz")
        outfile = np.load(outfilename)
        self.train_predictions = outfile['train_predictions']
        self.test_predictions = outfile['test_predictions']
        self.hypotheses = outfile['hypotheses']
        
    def _combine_val(self):
        for x in range(1,self.pb.xval_no+1):
            outfilename = str(self.pb.wd + 
                              "out_"+str(self.pb.conf_num)+"_"+str(x)+".npz")
            outfile = np.load(outfilename)
            valInd = (self.pb.xval_indices == x)
            v = outfile['val_predictions']
            self.validation_predictions[valInd,:] = v

    def _create_dir(self):
        """ Create directory of a portable final classifier """
        self.out_fp = str(self.pb.wd + 
                                  'out_'+str(self.pb.conf_num) + '/')
        if not os.path.exists(self.out_fp):
            os.makedirs(self.out_fp)
    
    def _get_feature_def(self,cursor,rowid):
        s = 'SELECT * FROM features WHERE rowid='+str(rowid+1)
        cursor.execute(s)
        return list(cursor.fetchone())
    
    def dump(self):
        """ 
        Dumps in the final classifier directory a set of information 
        that can be used by any computer later on to reproduce the graphs.
        Useful for running on servers that do not have a screen for immediate
        graph presentation 
        """
        dump_filename = self.out_fp + "dump.npz"
        meta = np.array([self.pb.wd,
                         self.pb.algorithm,
                         self.pb.rounds,
                         self.pb.total_exam_no,
                         self.pb.test_exam_no,
                         self.pb.testEN,
                         self.pb.xval_no,
                         self.pb.xvalEN]) 
        if self.pb.testEN:
            if self.pb.xvalEN:
                np.savez(dump_filename,
                         meta = meta,
                         train_predictions = self.train_predictions,
                         train_label = self.train_label,
                         test_predictions = self.test_predictions,
                         test_label = self.test_label,
                         validation_predictions = self.validation_predictions,
                         xval_indices = self.pb.xval_indices)
            else:
                np.savez(dump_filename,
                         meta = meta,
                         train_predictions = self.train_predictions,
                         train_label = self.train_label,
                         test_predictions = self.test_predictions,
                         test_label = self.test_label,
                         )
        else:
            if self.pb.xvalEN:
                np.savez(dump_filename,
                         meta = meta,
                         train_predictions = self.train_predictions,
                         train_label = self.train_label,
                         validation_predictions = self.validation_predictions,
                         xval_indices = self.pb.xval_indices)                
            else:
                np.savez(dump_filename,
                         meta = meta,
                         train_predictions = self.train_predictions,
                         train_label = self.train_label,
                         )

    def plot(self):
        """ 
        Decide which plots to present based on the configuration 
        file in use and plot them
        """
        plot_data(working_dir = self.pb.wd,
                  alg = self.pb.algorithm, 
                  rounds = self.pb.rounds, 
                  train_exam_no = self.pb.total_exam_no, 
                  test_exam_no = self.pb.test_exam_no, 
                  testEN = self.pb.testEN, 
                  xval_no = self.pb.xval_no, 
                  xvalEN = self.pb.xvalEN,
                  train_predictions = self.train_predictions, 
                  train_label = self.train_label, 
                  test_predictions = self.test_predictions, 
                  test_label = self.test_label, 
                  validation_predictions = self.validation_predictions, 
                  xval_indices = self.pb.xval_indices, 
                  filename = None,
                  basepath = self.out_fp,
                  only_save = False)
        
    def report(self):
        plot_data(working_dir = self.pb.wd,
                  alg = self.pb.algorithm, 
                  rounds = self.pb.rounds, 
                  train_exam_no = self.pb.total_exam_no, 
                  test_exam_no = self.pb.test_exam_no, 
                  testEN = self.pb.testEN, 
                  xval_no = self.pb.xval_no, 
                  xvalEN = self.pb.xvalEN,
                  train_predictions = self.train_predictions, 
                  train_label = self.train_label, 
                  test_predictions = self.test_predictions, 
                  test_label = self.test_label, 
                  validation_predictions = self.validation_predictions, 
                  xval_indices = self.pb.xval_indices, 
                  filename = None,
                  basepath = self.out_fp,
                  only_save = True)
    
    def create_exec(self):
        """ Create a directory of files for a stand-alone final classifier """
        
        """
        Modify hypotheses fn_def file paths before storing them to the
        final_classifier directory, so that previous structure is not visible 
        in portable file
        """
        conn = sqlite3.connect(self.pb.feature_db)
        cursor = conn.cursor()        
        
        for h_ind in np.arange(len(self.hypotheses)):
            h = self.hypotheses[h_ind]
            offset = self.pb.partition[h['rnk']]
            fn_def = self._get_feature_def(cursor,offset + h['d1'])
            head,tail = os.path.split(fn_def[0])
            root,ext =  os.path.splitext(tail)
            fn_def[0] = './'+root+'.py'
            h['fn_def'] = fn_def
            self.hypotheses[h_ind] = h
        conn.close()
        
        """Save hypotheses to final_classifier"""
        hypotheses_path = self.out_fp + 'hypotheses.npy'
        np.save(hypotheses_path, self.hypotheses)
        
        """Copy user blueprintd functions to final_classifier"""
        for feature_fp in self.pb.factory_files:
            if feature_fp == 'default':
                src_path = os.path.abspath(factory.__file__).replace('.pyc','.py')
                feature_fp = "factory.py"
            else:
                src_path = self.pb.wd + feature_fp
            dst_path = self.out_fp + feature_fp
            shutil.copyfile(src_path,dst_path)
        
        """Copy final_classifier script file"""
        final_classifier_path = self.out_fp + 'final_classifier.py'
        shutil.copyfile(os.path.abspath(final_classifier_script.__file__).replace('.pyc','.py'),
                        final_classifier_path)

    def clean(self):
        """ 
        Clean up unncessary intermediate files from working directory 
        to avoid polluting disk 
        """
        for filename in glob.glob(self.pb.wd+'*.npz'):
            os.remove(filename)

        for filename in glob.glob(self.pb.wd+'*.npy'):
            os.remove(filename)

        for filename in glob.glob(self.pb.wd+'*.h5'):
            os.remove(filename)
            
        for filename in glob.glob(self.pb.wd+'*.db'):
            os.remove(filename)

    def run(self):
        if self.pb.show_plots == 'y':
            self.plot()
        else:
            self.report()
        self.dump()
        self.create_exec()
        self.clean()

def plot_data(working_dir = None,
              alg = None, 
              rounds = None, 
              train_exam_no = None, 
              test_exam_no = None, 
              testEN = None, 
              xval_no = None, 
              xvalEN = None,
              train_predictions = None, 
              train_label = None, 
              test_predictions = None, 
              test_label = None, 
              validation_predictions = None, 
              xval_indices = None, 
              filename = None,
              basepath = None,
              only_save = False):
    """ 
    Stand-alone method that can be used to plot the data. 
    It takes information needed either from a file, or a reporting object 
    every input is None by default, since some calls only require filename and
    some calls require everything but a filename.
    """
    
    """ 
    Decide which plots to present based on the configuration file 
    in use and plot them 
    """
    if filename:
        f = np.load(filename)
        (working_dir, alg, rounds, train_exam_no, 
         test_exam_no, testEN, xval_no, xvalEN) = f['meta']
        rounds = int(rounds)
        train_exam_no = int(train_exam_no)
        train_predictions = f['train_predictions']
        train_label = f['train_label']
        if testEN:
            test_exam_no = int(test_exam_no)
            test_predictions = f['test_predictions']
            test_label = f['test_label']
        if xvalEN:
            validation_predictions = f['validation_predictions']
        if not basepath:
            basepath = os.path.realpath("./")
    elif not basepath:
        basepath = working_dir
            
    if testEN:
        if xvalEN:
            """1) ROC plot based on combined validation error"""
            last_round = validation_predictions[:,-1]
            z1 = zip(train_label, last_round)
            roc1 = pyroc.ROCData(z1)
            print("Info : Confusion matrix for combined validation "
                    +"error with zero threshold :")    
            print(roc1.confusion_matrix(0))        

            """2) ROC plot based on testing error"""
            last_round = test_predictions[:,-1]
            z2 = zip(test_label, last_round)
            roc2 = pyroc.ROCData(z2)
            print("Info : Confusion matrix for testing "
                    +"error with zero threshold :")    
            print(roc2.confusion_matrix(0))        
            
            pyroc.plot_multiple_roc(filename = basepath + "roc.tif",
                                    rocList = (roc1, roc2), 
                                    title='ROC Curve',
                                    labels = ("validation error curve","testing error curve"),
                                    only_save = only_save)
            
            """3) Plot training error against number of rounds"""
            training_error = np.zeros(shape = [rounds,1] , dtype = "float32")
            test_error = np.zeros(shape = [rounds,1] , dtype = "float32")
            val_error = np.zeros(shape = [rounds,1] , dtype = "float32")
            for k in range(rounds):
                training_error[k] = sum((2*train_label-1) 
                                        != np.sign(train_predictions[:, k])
                                        )/np.float32(train_exam_no)
                val_error[k] = sum((2*train_label-1) 
                                   != np.sign(validation_predictions[:, k])
                                   )/np.float32(train_exam_no) 
                test_error[k] = sum((2*test_label-1) 
                                    != np.sign(test_predictions[:, k])
                                    )/np.float32(test_exam_no)

            training_err = sum((2*train_label-1) 
                                 != np.sign(train_predictions[:, -1])
                                 )/np.float32(train_exam_no)
            print ("Info : Training Error of the final classifier : "+ 
                   str(training_err))
            val_err = sum((2*train_label-1) 
                            != np.sign(validation_predictions[:,-1])
                            )/np.float32(train_exam_no)
            print ("Info : Validation Error of the final classifier : "+ 
                   str(val_err))
            test_err = sum((2*test_label-1) 
                             != np.sign(test_predictions[:,-1])
                             )/np.float32(test_exam_no)            
            print ("Info : Testing Error of the final classifier : "+ 
                   str(test_err))
            
            """Plotting using draw since it's non-blocking"""
            if not only_save:
                plt.figure()
                plt.subplot(2,2,1)
                plt.plot(range(rounds), training_error)
                plt.xlabel('number of rounds')
                plt.ylabel('training error')
                plt.title('Training Error vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(2,2,2)
                plt.plot(range(rounds), val_error)
                plt.xlabel('number of rounds')
                plt.ylabel('validation error')
                plt.title('Validation Error vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(2,2,3)
                plt.plot(range(rounds), test_error)
                plt.xlabel('number of rounds')
                plt.ylabel('testing error')
                plt.title('Testing Error vs. Number of Rounds')             
                plt.draw()                    
                """show is blocking, and is called after all graphs are created"""
                plt.show()

            plt.clf()
            plt.plot(range(rounds), training_error)
            plt.xlabel('number of rounds')
            plt.ylabel('training error')
            plt.title('Training Error vs. Number of Rounds')
            plt.savefig(filename = basepath + "training_err.tif")

            """4) Plot test error against number of rounds"""
            plt.clf()
            plt.plot(range(rounds), val_error)
            plt.xlabel('number of rounds')
            plt.ylabel('validation error')
            plt.title('Validation Error vs. Number of Rounds')
            plt.savefig(filename = basepath + "validation_err.tif")
            
            """5) Plot test error against number of rounds"""
            plt.clf()
            plt.plot(range(rounds), test_error)
            plt.xlabel('number of rounds')
            plt.ylabel('testing error')
            plt.title('Testing Error vs. Number of Rounds')             
            plt.savefig(filename = basepath + "testing_err.tif")
        else:
            """1) ROC plot based on testing error"""
            last_round = test_predictions[:,-1]
            z = zip(test_label, last_round)
            roc = pyroc.ROCData(z)
            roc.plot(filename = basepath+"roc.tif",
                     title='Testing Error ROC Curve',
                     only_save = only_save)
            print("Info : Confusion matrix for combined validation "
                    +"error with zero threshold :")  
            print(roc.confusion_matrix(0))

            """2) Plot training error against number of rounds"""
            training_error = np.zeros(shape = [rounds,1] , dtype = "float32")
            test_error = np.zeros(shape = [rounds,1] , dtype = "float32")
            for k in range(rounds):
                training_error[k] = sum((2*train_label-1) 
                                        != np.sign(train_predictions[:, k])
                                        )/np.float32(train_exam_no)
                test_error[k] = sum((2*test_label-1) 
                                    != np.sign(test_predictions[:, k])
                                    )/np.float32(test_exam_no)
            
            training_err = sum((2*train_label-1) 
                                 != np.sign(train_predictions[:, -1])
                                 )/np.float32(train_exam_no)
            print ("Info : Training Error of the final classifier : "+ 
                   str(training_err))
            test_err = sum((2*test_label-1) 
                             != np.sign(test_predictions[:,-1])
                             )/np.float32(test_exam_no)            
            print ("Info : Testing Error of the final classifier : "+ 
                   str(test_err))
            
            if not only_save:
                plt.figure()
                plt.subplot(1,2,1)
                plt.plot(range(rounds), training_error)
                plt.xlabel('number of rounds')
                plt.ylabel('training error')
                plt.title('Training Error vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(1,2,2)
                plt.plot(range(rounds), test_error)
                plt.xlabel('number of rounds')
                plt.ylabel('testing error')
                plt.title('Testing Error vs. Number of Rounds')             
                plt.draw()                    
                """show is blocking, and is called after all graphs are created"""
                plt.show()
            plt.clf()
            plt.plot(range(rounds), training_error)
            plt.xlabel('number of rounds')
            plt.ylabel('training error')
            plt.title('Training Error vs. Number of Rounds')
            plt.savefig(filename = basepath + "training_err.tif")

            """5) Plot test error against number of rounds"""
            plt.clf()
            plt.plot(range(rounds), test_error)
            plt.xlabel('number of rounds')
            plt.ylabel('testing error')
            plt.title('Testing Error vs. Number of Rounds')             
            plt.savefig(filename = basepath + "testing_err.tif")

    else:
        if xvalEN:
            """1) ROC plot based on combined validation error"""
            last_round = validation_predictions[:,-1]
            z = zip(train_label, last_round)
            roc = pyroc.ROCData(z)
            roc.plot(filename = basepath+"roc.tif",
                     title='Validation Error ROC Curve',
                     only_save = only_save)
            print("Info : Confusion matrix for combined validation "
                    +"error with zero threshold :")  
            print(roc.confusion_matrix(0))                 

            """2) Plot training error against number of rounds"""
            training_error = np.zeros(shape = [rounds,1] , dtype = "float32")
            val_error = np.zeros(shape = [rounds,1] , dtype = "float32")
            for k in range(rounds):
                training_error[k] = sum((2*train_label-1) 
                                        != np.sign(train_predictions[:, k])
                                        )/np.float32(train_exam_no)
                val_error[k] = sum((2*train_label-1) 
                                   != np.sign(validation_predictions[:, k])
                                   )/np.float32(train_exam_no) 
            
            training_err = sum((2*train_label-1) 
                                 != np.sign(train_predictions[:, -1])
                                 )/np.float32(train_exam_no)
            print ("Info : Training Error of the final classifier : "+ 
                   str(training_err))
            val_err = sum((2*train_label-1) 
                            != np.sign(validation_predictions[:,-1])
                            )/np.float32(train_exam_no)
            print ("Info : Validation Error of the final classifier : "+ 
                   str(val_err))
            
            if not only_save:
                plt.figure()
                plt.subplot(1,2,1)
                plt.plot(range(rounds), training_error)
                plt.xlabel('number of rounds')
                plt.ylabel('training error')
                plt.title('Training Error vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(1,2,2)
                plt.plot(range(rounds), val_error)
                plt.xlabel('number of rounds')
                plt.ylabel('validation error')
                plt.title('Validation Error vs. Number of Rounds')
                plt.draw()
                plt.show()
            plt.clf()
            plt.plot(range(rounds), training_error)
            plt.xlabel('number of rounds')
            plt.ylabel('training error')
            plt.title('Training Error vs. Number of Rounds')
            plt.savefig(filename = basepath + "training_err.tif")

            """4) Plot test error against number of rounds"""
            plt.clf()
            plt.plot(range(rounds), val_error)
            plt.xlabel('number of rounds')
            plt.ylabel('validation error')
            plt.title('Validation Error vs. Number of Rounds')
            plt.savefig(filename = basepath + "validation_err.tif")

        else:
            """1) Plot training error against number of rounds"""
            training_error = np.zeros(shape = [rounds,1] , dtype = "float32")
            for k in range(rounds):
                training_error[k] = sum((2*train_label-1) 
                                        != np.sign(train_predictions[:, k])
                                        )/np.float32(train_exam_no)
            
            training_err = sum((2*train_label-1) 
                                 != np.sign(train_predictions[:, -1])
                                 )/np.float32(train_exam_no)
            print ("Info : Training Error of the final classifier : "+ 
                   str(training_err))
            
            """Plotting using draw since it's non-blocking"""
            plt.figure()
            plt.plot(range(rounds), training_error)
            plt.xlabel('number of rounds')
            plt.ylabel('training error')
            plt.title('Training Error vs. Number of Rounds')
            plt.savefig(filename = basepath + "training_err.tif")
            if not only_save:
                plt.show()