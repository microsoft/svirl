from setuptools import setup, find_packages, find_namespace_packages


def readme():
    with open('README.md') as f:
        return f.read()


# https://docs.python.org/3/distutils/setupscript.html#meta-data
class svirl_metadata():

    name = 'svirl'
    version = '1.0' # Format: major.minor.build.revision
    description = 'GPU accelerated Ginzburg-Landau equations solver'
    long_description = readme() 

    provides = []
    requires = [
        'python>=3.0',
        'pycuda>=2018.1',
        'numpy>=1.15',
        'scipy>=1.1',
        'matplotlib>=3.0',
        'PIL>=1.1.6',
        'cmocean>=1.2',
    ]

    maintainer = 'Shriram Jagannathan, Ivan Sadovskyy'
    maintainer_email = 'svirl@outlook.com'

    url = 'https://github.com/microsoft/svirl'
    license = 'MIT'

    # This is a required input
    packages = find_namespace_packages(include=['svirl*'])

    # https://pypi.org/classifiers/
    classifiers = [
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        # 'Topic :: Software Development',
        'Intended Audience :: Science/Research',
        # 'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ] 

    keywords = [
        'Superconductivity',
        'Vortex dynamics',
        'Ginzburg-Landau equations',
        'GL equations'
        'Time-dependent Ginzburg-Landau',
        'TDGL',
        'GL free energy minimization'
        'Non-linear conjugate gradient method',
        'Scientific computing',
        'Non-linear PDE',
        'GPU',
        'CUDA',
    ]


svirl = svirl_metadata()

# https://setuptools.readthedocs.io/en/latest/setuptools.html#developer-s-guide
setup(
    name                 = svirl.name,
    version              = svirl.version,
    description          = svirl.description,
    license              = svirl.license,
    packages             = svirl.packages,
    maintainer           = svirl.maintainer,
    maintainer_email     = svirl.maintainer_email,
    include_package_data = True,
    zip_safe             = False,
)
