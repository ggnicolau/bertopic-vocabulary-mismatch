from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='We used BERTopic + BM25 to conduct experiments for vocabulary mismatch and clustering data for information retrieval. We chose entropy as our metric.',
    author='Guilherme Giuliano Nicolau',
    license='MIT',
)
