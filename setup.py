from setuptools import setup, find_packages

setup(
    name='pyccsn',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ],
    author='Jungyoung Kim',
    description='Computational neuroscience tools for spectrum, filtering, correlation, and signal analysis',
    long_description=open('README.md', encoding="UTF8").read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)