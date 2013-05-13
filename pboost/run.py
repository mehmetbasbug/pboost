import os, sys, argparse
from mpi4py import MPI
import traceback

try:
    from pboost.environment.pb import PBoost,PBoostMPI
except ImportError:
    """
    If pboost is not installed append parent directory of 
    run.py into PYTHONPATH and try again
    """
    sys.path.append(os.path.dirname(
                    os.path.dirname(os.path.realpath(__file__)))
                    )
    from pboost.environment.pb import PBoostMPI
    pass

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
                description='Runs pboost for the specified configuration ')
    
    argparser.add_argument('--conf_path','-cp',
                           type=str,
                           default='~/pboost/configurations.cfg',
                           help='Path to the configuration file')
    argparser.add_argument('conf_intervals',
                           metavar='ci',
                           type=str, 
                           nargs='+',
                           help='Configuration numbers to process, either a single number or an interval i.e. 2 3 5 or 2-10 17')
    argparser.add_argument('--debug','-db',
                           type=str,
                           default='n',
                           help='Enable debug mode [y/n]')
    args = argparser.parse_args()
    conf_path = os.path.realpath(os.path.expanduser(args.conf_path))
    debugEN = args.debug == 'y'
    comm = MPI.COMM_WORLD
    conf_nums = list()
    for conf_interval in args.conf_intervals:
        conf_interval = conf_interval.split('-')
        if len(conf_interval) == 1:
            conf_nums.append(int(conf_interval[0]))
        elif len(conf_interval) == 2:
            for conf_num in range(int(conf_interval[0]),int(conf_interval[1])+1):
                conf_nums.append(int(conf_num))
        else:
            raise Exception("Inappropriate format for configuration numbers. See help")
    for conf_num in conf_nums:
        try:
            if comm.Get_size()>1:
                pb = PBoostMPI(comm = comm, 
                            conf_num = conf_num, 
                            conf_path = conf_path,
                            debugEN = debugEN
                            )
            else:
                pb = PBoost(conf_num = conf_num, 
                            conf_path = conf_path,
                            debugEN = debugEN
                            )
            pb.run()
            pb = None # Explicitly remove the object for garbage collection
        except Exception as e:
            print traceback.format_exc()
            comm.Abort()
            sys.exit()