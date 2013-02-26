import numpy as np
import argparse,h5py
from pboost.feature.extract import Feature

def main(data_fp):
    try:
        data_file = h5py.File(data_fp,'r')
        data = data_file['data']
    except:
        data = np.load(data_fp)
        pass
    
    hypotheses = np.load('hypotheses.npy')
    dump_f = np.load('dump.npz')
    (working_dir, alg, rounds, train_exam_no, 
     test_exam_no, testEN, xval_no, xvalEN) = dump_f['meta']

    result = np.zeros((data.shape[0],))
    for k in range(hypotheses.shape[0]):
        h = hypotheses[k]
        feat = Feature(data = data,
                       feature_def = h['fn_def'])
        tmp = feat.apply(params = h['fn_def'][2])
        if alg == 'conf-rated':
            s1 = np.int16([tmp <= h['v']])*h['c0']
            s2 = np.int16([tmp > h['v']])*h['c1']
        elif alg == 'adaboost':
            s1 = np.int16([tmp <= h['v']])*np.sign(h['c0'])
            s2 = np.int16([tmp > h['v']])*np.sign(h['c1'])
        else:
            raise Exception("Unknown Boosting Algorithm")
        result = result + s1[0] + s2[0]
    if data_file:
        data_file.close()
    prediction = (np.sign(result)+1)/2
    print 'Final classifier prediction =', prediction
    np.save('predictions.npy',prediction)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
                description='Making prediction with Final Classifier')
    
    argparser.add_argument('data_fp', 
                           metavar='dfp', 
                           type=str,
                           help='Path to the data file')
    args = argparser.parse_args()
    main(args.data_fp)
