import time
import os

def check_file(path_to_file,sleep_time):
    """
    Called by other modules to check that writing to disk is completed
    """
    while True:
        try:
            with open(path_to_file) as f: pass
        except IOError as e:
            time.sleep(sleep_time)
            continue
        break
    size1 = 1;
    size2 = 2;
    while size1 != size2:
        size1 = max(os.path.getsize(path_to_file),1);
        time.sleep(sleep_time*10);
        size2 = max(os.path.getsize(path_to_file),2);
