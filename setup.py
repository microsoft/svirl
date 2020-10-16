from setuptools import setup, find_packages, find_namespace_packages

def readme():
    with open('README.md') as f:
        return f.read()

# https://docs.python.org/3/distutils/setupscript.html#meta-data
class svirl_metadata():

    name = 'svirl'
    version = '1.0' # Format: major.minor.build.revision
    description = '''GPU accelerated time-dependent Ginzburg-Landau solver''' 
    long_description = readme() 

    provides = []
    requires = ["python>=3.0", 
                "pycuda", 
                "numpy",
                "scipy"
                "matplotlib",
                ]  

    maintainer = 'Ivan Sadovskyy, Shriram Jagannathan'
    maintainer_email = 'abc@email.com'

    url = 'https://github.com/microsoft/svirl'
    license = 'MIT'

    # this is a required input
    packages    = find_namespace_packages(include=["svirl*"])

    # https://pypi.org/classifiers/
    classifiers = [
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            ] 

    keywords = ['Time Dependent Ginzburg-Landau equations',
                'Non-linear Conjugate Gradient method',
                'Numerical Optimization', 
                'Non-linear PDE',
                'CUDA',
                'GPU',
                'Superconductivity',
                'Scientific Computing', 
                'Computational Science', 
                ]


svirl = svirl_metadata()

#https://setuptools.readthedocs.io/en/latest/setuptools.html#developer-s-guide
setup(
      name         = svirl.name,
      version      = svirl.version,
      description  = svirl.description,
      license      = svirl.license,
      packages     = svirl.packages,
      include_package_data = True,

      zip_safe = False,
      )
