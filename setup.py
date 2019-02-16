import os
from glob import glob
from setuptools import setup
from setuptools.config import read_configuration

config = read_configuration('setup.cfg')
config_dict = {}

for section in config:
    for k in config[section]:
        config_dict[k] = config[section][k]

if os.path.exists('scripts'):
    config_dict['scripts'] = glob(os.path.join('scripts', '*'))

setup(**config_dict)
