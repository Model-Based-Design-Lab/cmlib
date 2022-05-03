from setuptools import setup, find_packages

setup(
    name='markovchains',
    version='1.0',
    description=
    'Package to support operations on discrete time Markov chains',
    url='https://computationalmodeling.info',
    author='Marc Geilen',
    author_email='m.c.w.geilen@tue.nl',
    license='MIT',
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        'python-graph-core',
        'numpy',
        'textx',
        'matplotlib',
        'pandas',
        'statistics',
    ],
    entry_points={"console_scripts": ['markovchains = markovchains.utils.commandline:main']},
    tests_require=[
        'python-graph-core',
        'numpy',
        'textx',
        'matplotlib',
        'pandas',
        'pytest',
        'statistics',
    ],
)
