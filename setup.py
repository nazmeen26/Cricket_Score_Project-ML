from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements  
    

setup(
name='T20_CRICKET_PROJECT',
version='0.0.1',
auther='nazmeen',
auther_email='khannazmeen7860@gmail.com',
packages=find_packages(),
unstall_requirements=get_requirements('requirements.txt')


)