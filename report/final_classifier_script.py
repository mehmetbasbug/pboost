from os import path
from json import loads
from numpy import sign, load
import sys

def main(example_data):
    data = load(str(example_data))
    hypotheses = load('hypotheses.npy')

    result = 0.0
    for k in range(hypotheses.shape[0]):
        print result
        head,tail = path.split(hypotheses[k][0]['fn_def'][0])
        root,ext = path.splitext(tail)
        m = __import__(root)
        bhv_class = getattr(m,hypotheses[k][0]['fn_def'][1])
        bhv_obj = bhv_class(data)
        args = loads(hypotheses[k][0]['fn_def'][2])
        tmp = bhv_obj.behavior(*args)
        if tmp > hypotheses[k][0]['v']:
            result += hypotheses[k][0]['c1']
        else:    
            result += hypotheses[k][0]['c0']    

    print 'Final classifier result =', int(sign(result)+1)/2

if __name__ == '__main__':
    example_data = sys.argv[1]
    main(example_data = example_data)
