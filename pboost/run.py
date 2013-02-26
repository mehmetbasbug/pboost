import os, sys, argparse
from mpi4py import MPI
try:
    from pboost.environment.pb import PBoost
except ImportError:
    """
    If parallel_boosting is not installed append parent directory of 
    pboost.py into PYTHONPATH
    """
    sys.path.append(os.path.dirname(
                    os.path.dirname(os.path.realpath(__file__)))
                    )
    from pboost.environment.pb import PBoost
    pass

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
                description='Runs pboost for the specified configuration ')
    
    argparser.add_argument('conf_path', 
                           metavar='cp', 
                           type=str,
                           help='Path to the configuration file')
    argparser.add_argument('conf_nums', 
                           metavar='cn', 
                           type=int, 
                           nargs='+',
                           help='Configuration numbers to process')
    args = argparser.parse_args()
    conf_path = os.path.realpath(args.conf_path)
    comm = MPI.COMM_WORLD
    for conf_num in args.conf_nums:
        try:
            pb = PBoost(comm = comm, 
                        conf_num = conf_num, 
                        conf_path = conf_path, 
                        )
            pb.run()
            pb = None # Explicitly remove the object for garbage collection
        except Exception as e:
            print e
            comm.Abort()
            sys.exit()