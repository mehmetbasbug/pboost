import sys, os
import argparse

try:
    from pboost.report.reporter import plot_data,report_results
except ImportError:
    """
    If pboost is not installed append parent directory of 
    run.py into PYTHONPATH and try again
    """
    sys.path.append(os.path.dirname(
                    os.path.dirname(os.path.realpath(__file__)))
                    )
    from pboost.report.reporter import plot_data,report_results
    pass

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
                description='Shows the plots for given pboost result file ')
    
    argparser.add_argument('filepaths',
                           metavar='fp', 
                           type=str,
                           nargs='+',
                           help='Path to the result file')
    argparser.add_argument('--report_only','-ro', 
                           type=str,
                           default='n',
                           help='y : report only n : show plots(default)')
    
    args = argparser.parse_args()
    for filepath in args.filepaths:
        filepath = os.path.realpath(filepath)
        if args.report_only == 'y':
            report_results(filename = filepath)
        else:
            plot_data(filename = filepath)