from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['asee2_core'
             ],
    package_dir={'asee2_core': 'asee2_core'
                },
)
setup(**setup_args)