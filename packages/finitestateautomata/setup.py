from setuptools import setup, find_packages

setup(
    name='finitestateautomata',
    version='1.0',
    description=
    'Package to support operations on finite state automata',
    url='https://github.com/Model-Based-Design-Lab/cmlib/tree/main/packages/finitestateautomata',
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
    entry_points={"console_scripts": ['finitestateautomata = finitestateautomata.utils.commandline:main']},
    tests_require=['pytest'],
)
