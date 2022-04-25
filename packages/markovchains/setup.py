from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

class NoseTestCommand(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # Run nose ensuring that argv simulates running nosetests directly
        import nose
        nose.run_exit(argv=['nosetests'])

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
        'nose',
        'statistics',
    ],
    entry_points={"console_scripts": ['markovchains = markovchains.utils.commandline:main']},
    test_suite='nose.collector',
    tests_require=[
        'python-graph-core',
        'numpy',
        'textx',
        'matplotlib',
        'pandas',
        'nose',
        'statistics',
    ],
)
