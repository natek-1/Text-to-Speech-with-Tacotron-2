from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirments(file_path:str)->List[str]:
    '''
    Using a given file path it returns all the packages that are in the given file

    file_path: path to the file with the requirments 

    returns: list object with the packages as strings
    '''
    requires = []
    with open(file_path, "r") as file:
        requires = file.readlines()
        requires = [req.replace("\n", "") for req in requires]

    if HYPHEN_E_DOT in requires:
        requires.remove(HYPHEN_E_DOT)

    return requires

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

AUTHOR_USER_NAME="natek-1"
REPO_NAME="Text-to-Speech-with-Tacotron-2"
package_name = "tts"


setup(
    name=package_name,
    version="0.0.1",
    author="Nathan Kuissi",
    author_email="nategabrielk@icloud.com",
    description="A python package for training the Tacotron 2 model from scratch using pytorch for text to speech",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    long_description=long_description,
    long_description_content="text/markdown",
    install_requires=get_requirments("requirements.txt"),
    packages=find_packages()
)