"""
Configuration parser module
Configuration files are specified in INI format, with different "sections" corresponding
to particular configuration numbers.
"""

import numpy as np
import ConfigParser
import os

def get_configuration(conf_num, conf_path = "./configurations.cfg"):
    """
    Parse the different fields of the configuration file, and 
    store them in a dictionary.
    Make sure the working directory is appended with '/' at the end.
    """
    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read(conf_path)
    section = "Configuration " + str(conf_num)
    try:
        options = config.options(section)
    except ConfigParser.NoSectionError:
        print ("Error : There is not any configuration with the specified"+
              " number. Please choose another number.")
    required_options = ('train_file','test_file','factory_files',
                        'algorithm','rounds','xval_no',
                        'working_dir','max_memory','show_plots'
                        )
    my_dict = dict()
    arbitrary = dict()
    for option in options:
        try:
            if option == 'working_dir': 
                my_dict['working_dir'] = os.path.realpath(
                             str(config.get(section,"working_dir")).strip()
                             )
            elif option in required_options:
                my_dict[option] = config.get(section, option)
            else:
                arbitrary[option] = config.get(section, option)
        except ConfigParser.NoOptionError:
            if option in required_options:
                print "Error : Configuration field " +option+ "is missing."
            else:
                continue
    my_dict['arbitrary'] = arbitrary
    
    # Convert to appropriate defaults
    if my_dict['train_file'] == None:
        raise Exception("``train_file`` input is mandatory")
    
    if my_dict['factory_files'] == None: 
        """Set default behavior"""
        my_dict['factory_files'] = ("default",)
    else:
        my_dict['factory_files'] = np.array(
                    [x.strip(' ') for x in my_dict['factory_files'].split(',')]
                    )

    if my_dict['algorithm'] == None:
        my_dict['algorithm'] = "conf-rated"

    if my_dict['rounds'] == None:
        my_dict['rounds'] = 20

    if my_dict['xval_no'] == None:
        my_dict['xval_no'] = 1

    if my_dict['working_dir'] == None:
        my_dict['working_dir'] = "./"

    if my_dict['max_memory'] == None:
        my_dict['max_memory'] = 2

    if my_dict['show_plots'] == None:
        my_dict['show_plots'] = 'y'


    # Convert strings to ints for numerical fields
    if my_dict['xval_no'] != None:
        try:
            my_dict['xval_no'] = int(my_dict['xval_no'])
            if my_dict['xval_no'] <= 0:
                raise Exception("Error : Configuration field ``xval_no`` "+
                                    "must be positive.")
        except ValueError:
            print "Configuration field ``xval_no`` must be an integer."

    if my_dict['rounds'] != None:
        try:
            my_dict['rounds'] = int(my_dict['rounds'])
            if my_dict['rounds'] <= 0:
                raise Exception("Error : Configuration field ``rounds`` "+
                                    "must be positive.")
        except ValueError:
            print "Configuration field ``rounds`` must be an integer."

    if my_dict['max_memory'] != None:
        try:
            my_dict['max_memory'] = float(my_dict['max_memory'])
            if my_dict['max_memory'] > 20 or my_dict['max_memory'] <= 0:
                raise Exception("Error : Configuration field `max_memory"+
                                   "must be specified in GBs, per core.")
        except ValueError:
            print ("Error : Configuration field ``max_memory`` must "+
                   "be an integer or floating point number.")

    """Make sure the working directory is appended with '/' at the end."""                
    if my_dict['working_dir'][-1] != '/':
        my_dict['working_dir'] = my_dict['working_dir'] + '/'
    
    return my_dict

