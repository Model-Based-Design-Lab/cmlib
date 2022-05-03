from setuptools import setup, find_packages

setup(
    name='dataflow',
    version='1.0',
    description=
    'Package to support operations on dataflow graphs',
    url='https://computationalmodeling.info',
    author='Marc Geilen',
    author_email='m.c.w.geilen@tue.nl',
    license='MIT',
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        'python-graph-core',
        'numpy',
        'textx'
    ],
    entry_points={"console_scripts": ['dataflow = dataflow.utils.commandline:main']},
    tests_require=['pytest'],
)
