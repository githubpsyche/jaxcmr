from setuptools import setup, find_packages
import os

def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as file:
        return [line.strip() for line in file if not line.startswith('#')]

setup(
    name='jaxcmr',
    version='0.1.0',
    description='the context maintenance and retrieval model implemented and evaluated using jax',
    author='Jordan Gunn',
    author_email='gunnjordanb@gmail.com',
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
