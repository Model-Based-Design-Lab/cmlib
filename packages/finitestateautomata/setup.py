from setuptools import setup, find_packages

setup(
    name='finitestateautomata',
    version='1.0',
    description=
    'Package to support operations on finite state automata',
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
        'nose',
    ],
    entry_points={"console_scripts": ['finitestateautomata = finitestateautomata.utils.commandline:main']},
    test_suite='nose.collector',
    tests_require=['nose'],
)
