from setuptools import setup, find_packages
import os,shutil,getpass
from pwd import getpwnam

readme_file = 'README'
version = __import__('pboost').get_version()

setup(name='pboost',
      version = version,
      description='Parallel Implementation of Boosting Algorithms with MPI.',
      long_description=open('README').read(),
      zip_safe=False,
      author='Mehmet Basbug',
      author_email='mbasbug@princeton.edu',
      url='https://github.com/mbasbug/pboost',
      license = "",
      packages = find_packages(),
      install_requires = [
        'numpy',
        'numexpr',
        'h5py',
        'mpi4py',
        'matplotlib',
        'ujson',
        'bitarray',
      ],
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Framework :: Buildout',
                   'Intended Audience :: Information Technology',
                   'License :: Free for non-commercial use',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   ],
      )

try:
    existing_path = os.path.abspath(__import__('pboost').__file__)
    existing_path,tail = os.path.split(existing_path)
    if os.name == "posix":
        pb_fp = os.path.join(os.path.expanduser('~'),'pboost')
        if not os.path.exists(pb_fp):
            os.mkdir(pb_fp)

        dest = os.path.join(pb_fp, "run.py")
        src = os.path.join(existing_path, "run.py")
        shutil.copyfile(src,dest)

        dest = os.path.join(pb_fp, "plot.py")
        src = os.path.join(existing_path, "plot.py")
        shutil.copyfile(src,dest)

        dest = os.path.join(pb_fp, "configurations.cfg")
        src = os.path.join(existing_path, "demo/configurations.cfg")
        shutil.copyfile(src,dest)

        dest = os.path.join(pb_fp, "demo")
        if not os.path.exists(dest):
            src = os.path.join(existing_path, "demo")
            shutil.copytree(src,dest)

        """If installing with sudo command change the permissions for pboost folder"""
        try:
            eff_username = os.getenv(key='SUDO_USER')
            if eff_username is not None:
                uid = getpwnam(eff_username).pw_uid
                gid = getpwnam(eff_username).pw_gid
                for root, dirs, files in os.walk(pb_fp):
                    os.chown(root, uid, gid)
                    for momo in dirs:
                        os.chown(os.path.join(root, momo), uid, gid)
                    for momo in files:
                        os.chown(os.path.join(root, momo), uid, gid)
        except:
            pass
except:
    pass
