import os, shutil, glob, inspect,sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
        self.train_error = None
        self.train_auc = None
        self.test_predictions = None
        self.test_error = None
        self.test_auc = None
        self.hypotheses = None
        self.validation_predictions = None
        self.validation_error = None
        self.validation_auc = None
        self.out_fp = None
        
        self._create_dir()
        
        if self.pb.xvalEN:
            self.validation_predictions = np.zeros(
                                          [self.pb.total_exam_no,self.pb.rounds],
                                          dtype="float32"
                                          )
            self._combine_val()
            
        if self.pb.testEN:
            self.test_label = pb.get_label(source = 'test')
            self._read_train_and_test()
            self._calculate_train()
            self._calculate_test()
        else:
            self._read_train()
            self._calculate_train()
            
        
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
        
    def _calculate_train(self):
        self.train_auc = np.zeros(self.pb.rounds)
        self.train_error = np.zeros(self.pb.rounds)
        self.validation_auc = np.zeros(self.pb.rounds)
        self.validation_error = np.zeros(self.pb.rounds)

        for round in range(self.pb.rounds):
            train_pred = self.train_predictions[:,round]
            z1 = zip(self.train_label, train_pred)
            roc1 = pyroc.ROCData(z1)
            self.train_auc[round] = roc1.auc()
            self.train_error[round] = sum((2*self.train_label-1) 
                                   != np.sign(self.train_predictions[:, round])
                                   )/np.float32(self.pb.total_exam_no) 
            if self.pb.xvalEN:
                val_pred = self.validation_predictions[:,round]
                z1 = zip(self.train_label, val_pred)
                roc1 = pyroc.ROCData(z1)
                self.validation_auc[round] = roc1.auc()
                self.validation_error[round] = sum((2*self.train_label-1) 
                                       != np.sign(self.validation_predictions[:, round])
                                       )/np.float32(self.pb.total_exam_no) 

    def _calculate_test(self):
        self.test_auc = np.zeros(self.pb.rounds)
        self.test_error = np.zeros(self.pb.rounds)
        for round in range(self.pb.rounds):
            test_pred = self.test_predictions[:,round]
            z1 = zip(self.test_label, test_pred)
            roc1 = pyroc.ROCData(z1)
            self.test_auc[round] = roc1.auc()
            self.test_error[round] = sum((2*self.test_label-1) 
                                   != np.sign(self.test_predictions[:, round])
                                   )/np.float32(self.pb.test_exam_no) 
        
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
                         train_error = self.train_error,
                         train_auc = self.train_auc,
                         train_label = self.train_label,
                         test_predictions = self.test_predictions,
                         test_error = self.test_error,
                         test_auc = self.test_auc,
                         test_label = self.test_label,
                         validation_predictions = self.validation_predictions,
                         validation_error = self.validation_error,
                         validation_auc = self.validation_auc,
                         xval_indices = self.pb.xval_indices)
            else:
                np.savez(dump_filename,
                         meta = meta,
                         train_predictions = self.train_predictions,
                         train_error = self.train_error,
                         train_auc = self.train_auc,
                         train_label = self.train_label,
                         test_predictions = self.test_predictions,
                         test_error = self.test_error,
                         test_auc = self.test_auc,
                         test_label = self.test_label,
                         )
        else:
            if self.pb.xvalEN:
                np.savez(dump_filename,
                         meta = meta,
                         train_predictions = self.train_predictions,
                         train_error = self.train_error,
                         train_auc = self.train_auc,
                         train_label = self.train_label,
                         validation_predictions = self.validation_predictions,
                         validation_error = self.validation_error,
                         validation_auc = self.validation_auc,
                         xval_indices = self.pb.xval_indices)                
            else:
                np.savez(dump_filename,
                         meta = meta,
                         train_predictions = self.train_predictions,
                         train_error = self.train_error,
                         train_auc = self.train_auc,
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
                  train_error = self.train_error,
                  train_auc = self.train_auc,
                  train_label = self.train_label, 
                  test_predictions = self.test_predictions,
                  test_error = self.test_error,
                  test_auc = self.test_auc,
                  test_label = self.test_label, 
                  validation_predictions = self.validation_predictions, 
                  validation_error = self.validation_error,
                  validation_auc = self.validation_auc,
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
                  train_error = self.train_error,
                  train_auc = self.train_auc, 
                  train_label = self.train_label, 
                  test_predictions = self.test_predictions,
                  test_error = self.test_error,
                  test_auc = self.test_auc,
                  test_label = self.test_label, 
                  validation_predictions = self.validation_predictions, 
                  validation_error = self.validation_error,
                  validation_auc = self.validation_auc,
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
        hypotheses = list()
        for h in self.hypotheses:
            for node in h.get_all_nodes():
                node.set_fn_def(cursor)
            d = h.to_dict()
            hypotheses.append(d)
        conn.close()
        
        """Save hypotheses to final_classifier"""
        hypotheses_path = self.out_fp + 'hypotheses.npy'
        np.save(hypotheses_path, hypotheses)
        
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

    def run(self):
        if self.pb.show_plots == 'y':
            self.plot()
        else:
            self.report()
        self.dump()
        self.create_exec()

def plot_data(working_dir = None,
              alg = None, 
              rounds = None, 
              train_exam_no = None, 
              test_exam_no = None, 
              testEN = None, 
              xval_no = None, 
              xvalEN = None,
              train_predictions = None, 
              train_error = None,
              train_auc = None,
              train_label = None, 
              test_predictions = None,
              test_error = None,
              test_auc = None, 
              test_label = None, 
              validation_predictions = None,
              validation_error = None,
              validation_auc = None, 
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
        train_error = f['train_error']
        train_auc = f['train_auc']
        train_label = f['train_label']
        if testEN:
            test_exam_no = int(test_exam_no)
            test_predictions = f['test_predictions']
            test_error = f['test_error']
            test_auc = f['test_auc']
            test_label = f['test_label']
        if xvalEN:
            validation_predictions = f['validation_predictions']
            validation_error = f['validation_error']
            validation_auc = f['validation_auc']
        if not basepath:
            filepath = os.path.realpath(os.path.expanduser(filename))
            head,tail = os.path.split(filepath)
            basepath = head + '/'
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
            
            pyroc.plot_multiple_roc(filename = basepath + "roc.png",
                                    rocList = (roc1, roc2), 
                                    title='ROC Curve',
                                    labels = ("validation error curve","testing error curve"),
                                    only_save = only_save)
            
            """3) Plot training error against number of rounds"""
            print ("Info : Training Error of the final classifier : "+ 
                   str(train_error[-1]))
            print ("Info : Validation Error of the final classifier : "+ 
                   str(validation_error[-1]))
            print ("Info : Testing Error of the final classifier : "+ 
                   str(test_error[-1]))
            
            """Plotting using draw since it's non-blocking"""
            if not only_save:
                plt.figure()
                plt.subplot(3,2,1)
                plt.plot(range(rounds), train_error)
                plt.xlabel('number of rounds')
                plt.ylabel('training error')
                plt.title('Training Error vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(3,2,2)
                plt.plot(range(rounds), train_auc)
                plt.xlabel('number of rounds')
                plt.ylabel('training auc')
                plt.title('Training Area Under ROC Curve vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(3,2,3)
                plt.plot(range(rounds), validation_error)
                plt.xlabel('number of rounds')
                plt.ylabel('validation error')
                plt.title('Validation Error vs. Number of Rounds')
                plt.draw()   

                plt.subplot(3,2,4)
                plt.plot(range(rounds), validation_auc)
                plt.xlabel('number of rounds')
                plt.ylabel('validation auc')
                plt.title('Validation Area Under ROC Curve vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(3,2,5)
                plt.plot(range(rounds), test_error)
                plt.xlabel('number of rounds')
                plt.ylabel('test error')
                plt.title('Test Error vs. Number of Rounds')
                plt.draw()   

                plt.subplot(3,2,6)
                plt.plot(range(rounds), test_auc)
                plt.xlabel('number of rounds')
                plt.ylabel('test auc')
                plt.title('Test Area Under ROC Curve vs. Number of Rounds')
                plt.draw()                    
                plt.show()

            plt.clf()
            plt.plot(range(rounds), train_error)
            plt.xlabel('number of rounds')
            plt.ylabel('training error')
            plt.title('Training Error vs. Number of Rounds')
            plt.draw()                    
            plt.savefig(filename = basepath + "training_err.png")

            plt.clf()
            plt.plot(range(rounds), train_auc)
            plt.xlabel('number of rounds')
            plt.ylabel('training auc')
            plt.title('Training Area Under ROC Curve vs. Number of Rounds')
            plt.draw()                    
            plt.savefig(filename = basepath + "training_auc.png")

            plt.clf()
            plt.plot(range(rounds), validation_error)
            plt.xlabel('number of rounds')
            plt.ylabel('validation error')
            plt.title('Validation Error vs. Number of Rounds')
            plt.draw()   
            plt.savefig(filename = basepath + "validation_err.png")

            plt.clf()
            plt.plot(range(rounds), validation_auc)
            plt.xlabel('number of rounds')
            plt.ylabel('validation auc')
            plt.title('Validation Area Under ROC Curve vs. Number of Rounds')
            plt.draw()                    
            plt.savefig(filename = basepath + "validation_auc.png")

            plt.clf()
            plt.plot(range(rounds), test_error)
            plt.xlabel('number of rounds')
            plt.ylabel('test error')
            plt.title('Test Error vs. Number of Rounds')
            plt.draw()   
            plt.savefig(filename = basepath + "test_err.png")
                        
            plt.clf()
            plt.plot(range(rounds), test_auc)
            plt.xlabel('number of rounds')
            plt.ylabel('test auc')
            plt.title('Test Area Under ROC Curve vs. Number of Rounds')
            plt.draw()   
            plt.savefig(filename = basepath + "test_auc.png")
            plt.close()
        else:
            last_round = test_predictions[:,-1]
            z = zip(test_label, last_round)
            roc = pyroc.ROCData(z)
            roc.plot(filename = basepath+"roc.png",
                     title='Testing Error ROC Curve',
                     only_save = only_save)
            print("Info : Confusion matrix for combined validation "
                    +"error with zero threshold :")  
            print(roc.confusion_matrix(0))

            print ("Info : Training Error of the final classifier : "+ 
                   str(train_error[-1]))
            print ("Info : Testing Error of the final classifier : "+ 
                   str(test_error[-1]))
            
            """Plotting using draw since it's non-blocking"""
            if not only_save:
                plt.figure()
                plt.subplot(2,2,1)
                plt.plot(range(rounds), train_error)
                plt.xlabel('number of rounds')
                plt.ylabel('training error')
                plt.title('Training Error vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(2,2,2)
                plt.plot(range(rounds), train_auc)
                plt.xlabel('number of rounds')
                plt.ylabel('training auc')
                plt.title('Training Area Under ROC Curve vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(2,2,3)
                plt.plot(range(rounds), test_error)
                plt.xlabel('number of rounds')
                plt.ylabel('test error')
                plt.title('Test Error vs. Number of Rounds')
                plt.draw()   

                plt.subplot(2,2,4)
                plt.plot(range(rounds), test_auc)
                plt.xlabel('number of rounds')
                plt.ylabel('test auc')
                plt.title('Test Area Under ROC Curve vs. Number of Rounds')
                plt.draw()                    
                plt.show()

            plt.clf()
            plt.plot(range(rounds), train_error)
            plt.xlabel('number of rounds')
            plt.ylabel('training error')
            plt.title('Training Error vs. Number of Rounds')
            plt.draw()                    
            plt.savefig(filename = basepath + "training_err.png")

            plt.clf()
            plt.plot(range(rounds), train_auc)
            plt.xlabel('number of rounds')
            plt.ylabel('training auc')
            plt.title('Training Area Under ROC Curve vs. Number of Rounds')
            plt.draw()                    
            plt.savefig(filename = basepath + "training_auc.png")

            plt.clf()
            plt.plot(range(rounds), test_error)
            plt.xlabel('number of rounds')
            plt.ylabel('test error')
            plt.title('Test Error vs. Number of Rounds')
            plt.draw()   
            plt.savefig(filename = basepath + "test_err.png")
                        
            plt.clf()
            plt.plot(range(rounds), test_auc)
            plt.xlabel('number of rounds')
            plt.ylabel('test auc')
            plt.title('Test Area Under ROC Curve vs. Number of Rounds')
            plt.draw()   
            plt.savefig(filename = basepath + "test_auc.png")
            plt.close()
    else:
        if xvalEN:
            """1) ROC plot based on combined validation error"""
            last_round = validation_predictions[:,-1]
            z1 = zip(train_label, last_round)
            roc = pyroc.ROCData(z1)
            print("Info : Confusion matrix for combined validation "
                    +"error with zero threshold :")    
            print(roc.confusion_matrix(0))    

            roc.plot(filename = basepath+"roc.png",
                     title='Validation ROC Curve',
                     only_save = only_save)
            
            """3) Plot training error against number of rounds"""
            print ("Info : Training Error of the final classifier : "+ 
                   str(train_error[-1]))
            print ("Info : Validation Error of the final classifier : "+ 
                   str(validation_error[-1]))
            
            """Plotting using draw since it's non-blocking"""
            if not only_save:
                plt.figure()
                plt.subplot(2,2,1)
                plt.plot(range(rounds), train_error)
                plt.xlabel('number of rounds')
                plt.ylabel('training error')
                plt.title('Training Error vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(2,2,2)
                plt.plot(range(rounds), train_auc)
                plt.xlabel('number of rounds')
                plt.ylabel('training auc')
                plt.title('Training Area Under ROC Curve vs. Number of Rounds')
                plt.draw()                    

                plt.subplot(2,2,3)
                plt.plot(range(rounds), validation_error)
                plt.xlabel('number of rounds')
                plt.ylabel('validation error')
                plt.title('Validation Error vs. Number of Rounds')
                plt.draw()   

                plt.subplot(2,2,4)
                plt.plot(range(rounds), validation_auc)
                plt.xlabel('number of rounds')
                plt.ylabel('validation auc')
                plt.title('Validation Area Under ROC Curve vs. Number of Rounds')
                plt.draw()                    
                plt.show()

            plt.clf()
            plt.plot(range(rounds), train_error)
            plt.xlabel('number of rounds')
            plt.ylabel('training error')
            plt.title('Training Error vs. Number of Rounds')
            plt.draw()                    
            plt.savefig(filename = basepath + "training_err.png")

            plt.clf()
            plt.plot(range(rounds), train_auc)
            plt.xlabel('number of rounds')
            plt.ylabel('training auc')
            plt.title('Training Area Under ROC Curve vs. Number of Rounds')
            plt.draw()                    
            plt.savefig(filename = basepath + "training_auc.png")

            plt.clf()
            plt.plot(range(rounds), validation_error)
            plt.xlabel('number of rounds')
            plt.ylabel('validation error')
            plt.title('Validation Error vs. Number of Rounds')
            plt.draw()   
            plt.savefig(filename = basepath + "validation_err.png")

            plt.clf()
            plt.plot(range(rounds), validation_auc)
            plt.xlabel('number of rounds')
            plt.ylabel('validation auc')
            plt.title('Validation Area Under ROC Curve vs. Number of Rounds')
            plt.draw()                    
            plt.savefig(filename = basepath + "validation_auc.png")

            plt.close()

        else:
            print ("Info : Training Error of the final classifier : "+ 
                   str(train_error[-1]))
            
            """Plotting using draw since it's non-blocking"""
            plt.figure()
            plt.plot(range(rounds), train_error)
            plt.xlabel('number of rounds')
            plt.ylabel('training error')
            plt.title('Training Error vs. Number of Rounds')
            plt.savefig(filename = basepath + "training_err.png")
            if not only_save:
                plt.show()
            plt.close()