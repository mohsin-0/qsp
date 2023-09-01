from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='qsp',
    description='implementation of different methods to prepare quantum states on quantum computer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mohsin-0/qsp',
    author='Mohsin Iqbal',
    author_email="mohsin.iqbal@cambridgequantum.com",
    license='Apache License 2.0',
    install_requires=[
        'quimb',
        'tensornetwork',
        'ncon',
        'tensorflow',
        'pymanopt',
        'qiskit',
        'qiskit-aer==0.11.2',
        'QGOpt',
    ],
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',      
        'Programming Language :: Python :: 3.9',
		'Programming Language :: Python :: 3.10',
		'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords='quantum state preparation tensor networks',
)
