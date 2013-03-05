from setuptools import setup, find_packages
import os,shutil

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
      include_package_data=True,
      package_data = {'pboost':['./demo/*.py','./demo/*.dat','./demo/*.cfg'],
                      },
      install_requires = [
        'numpy',
        'numexpr',
        'h5py',
        'mpi4py',
        'matplotlib',
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

    