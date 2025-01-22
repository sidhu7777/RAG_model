from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns the list of requirements.
    '''
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]
    return requirements

setup(
    name='RAG_Model_Development',
    version='0.0.1',
    author='vineeth',
    author_email='bvineeth76@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
