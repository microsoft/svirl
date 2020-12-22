from setuptools import setup, find_packages, find_namespace_packages


def readme():
    with open('README.md') as f:
        return f.read()


# https://docs.python.org/3/distutils/setupscript.html#meta-data
class svirl_metadata():

    from svirl import version 

    name = 'svirl'

    # Format: major.minor.revision
    version = version.version 

    description = "GPU accelerated Ginzburg-Landau equations solver"
    long_description = readme() 
    long_description_content_type = 'text/markdown'

    provides = []
    install_requires = [
        'pycuda>=2018.1',
        'numpy>=1.15',
        'scipy>=1.1',
        'matplotlib>=3.0',
        'cmocean>=1.2',
    ]

    python_requires='>=3.0'

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
    long_description     = svirl.long_description,
    long_description_content_type = svirl.long_description_content_type,
    license              = svirl.license,
    packages             = svirl.packages,
    install_requires     = svirl.install_requires,
    python_requires      = svirl.python_requires,
    maintainer           = svirl.maintainer,
    maintainer_email     = svirl.maintainer_email,
    include_package_data = True,
    zip_safe             = False,
)
