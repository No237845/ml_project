from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path: str) -> List[ str ]:
    '''This function will return the list of requirements'''
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='ml_project',
    version='0.0.1',
    author='Inno Kabore',
    email='kaboreinnocent93@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
