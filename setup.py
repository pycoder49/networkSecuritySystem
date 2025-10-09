from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]: 
    """
    This function will return the list of requirements
    """
    requirement_list: List[str] = []

    try:
        with open("requirements.txt", "r") as file:
            # reading lines
            lines = file.readlines()

            # processing linkes
            for line in lines:
                requirement = line.strip()

                # ignoring empty lines and -e .
                if requirement and requirement != "-e .":
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found. No dependencies will be installed.")
    
    return requirement_list


setup(
    name="Network Security",
    version="0.0.1",
    author="Aryan Ahuja",
    author_email="aryan-a@outlook.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)