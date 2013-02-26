from distutils.core import setup
from distutils.command.install_data import install_data
from distutils.command.install import INSTALL_SCHEMES
from distutils.sysconfig import get_python_lib
import os
import sys

#This script is based on the setup.py file provided in the following link:
#https://github.com/django/django/blob/master/setup.py

dublication = False
if "install" in sys.argv:
    #An explicit prefix "/usr/local" is also tried to catch the default path
    for lib_path in get_python_lib(), get_python_lib(prefix="/usr/local"):
        existing_path = os.path.abspath(os.path.join(lib_path, "pboost"))
        if os.path.exists(existing_path):
            dublication = True
            break

class osx_install_data(install_data):
    # On MacOS, the platform-specific lib dir is /System/Library/Framework/Python/.../
    # which is wrong. Python 2.5 supplied with MacOS 10.5 has an Apple-specific fix
    # for this in distutils.command.install_data#306. It fixes install_lib but not
    # install_data, which is why we roll our own install_data class.

    def finalize_options(self):
        # By the time finalize_options is called, install.install_lib is set to the
        # fixed directory, so we set the installdir to install_lib. The
        # install_data class uses ('install_data', 'install_dir') instead.
        self.set_undefined_options('install', ('install_lib', 'install_dir'))
        install_data.finalize_options(self)

if sys.platform == "darwin":
    cmdclasses = {'install_data': osx_install_data}
else:
    cmdclasses = {'install_data': install_data}

def fullsplit(path, result=None):
    """
    Split a pathname into components (the opposite of os.path.join) in a
    platform-neutral way.
    """
    if result is None:
        result = []
    head, tail = os.path.split(path)
    if head == '':
        return [tail] + result
    if head == path:
        return result
    return fullsplit(head, [tail] + result)

# Tell distutils not to put the data_files in platform-specific installation
# locations. See here for an explanation:
# http://groups.google.com/group/comp.lang.python/browse_thread/thread/35ec7b2fed36eaec/2105ee4d9e8042cb
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

#get the package names
packages, data_files = [], []
root_dir = os.path.dirname(__file__)
if root_dir != '':
    os.chdir(root_dir)
pck_dir = 'pboost'

for dirpath, dirnames, filenames in os.walk(pck_dir):
    for i, dirname in enumerate(dirnames):
        if dirname.startswith('.') or dirname == '__pycache__':     # ignored files
            del dirnames[i]
    if '__init__.py' in filenames:
        packages.append('.'.join(fullsplit(dirpath)))
    elif filenames:
        data_files.append([dirpath, [os.path.join(dirpath, f) for f in filenames]])
# Small hack for working with bdist_wininst.
# See http://mail.python.org/pipermail/distutils-sig/2004-August/004134.html
if len(sys.argv) > 1 and sys.argv[1] == 'bdist_wininst':
    for file_info in data_files:
        file_info[0] = '\\PURELIB\\%s' % file_info[0]

# Read the version number from the source code
version = __import__('pboost').get_version()

setup(name='pboost',
      version = version,
      description='Parallel Implementation of Boosting Algorithms with MPI.',
      long_description=open('README').read(),

      author='Mehmet Basbug',
      author_email='mbasbug@princeton.edu',
      url='https://github.com/mbasbug/pboost',
      license = "",
      packages = packages,
      cmdclass = cmdclasses,
      data_files = data_files,
      package_data = {'pboost': ['demo/*']},
      #package_data={'pboostz['latex/*.eps', 'latex/*.tex'
      #                           'html/*.html']},
      classifiers=['Development Status :: Beta',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Machine Learning']
      )

if dublication:
    sys.stderr.write("""

Warning: You have installed Parallel Boosting package on top of an already existing
installation.This may cause some problems. The recommended way
is to remove the package in the following path

%(existing_path)s

manually and re-install Parallel Boosting.

""" % { "existing_path": existing_path })
