import numpy as np
import argparse,h5py
from pboost.boost.decision_tree import Tree

def main(data_fp):
    try:
        data_file = h5py.File(data_fp,'r')
        data = data_file['data']
    except:
        data = np.load(data_fp)
        pass
    
    hypotheses = np.load('hypotheses.npy')
    result = np.zeros(data.shape[0])
    for d in hypotheses:
        h = Tree()
        h.from_dict(d)
        result = result + h.predict(data)
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
