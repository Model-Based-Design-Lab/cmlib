from setuptools import setup, find_packages

setup(
    name='modeltest',
    version='1.0',
    description=
    'Package to support testing of model functions using pytest',
    url='https://github.com/Model-Based-Design-Lab/cmlib/tree/main/packages/modeltest',
    author='Joep van Wanrooij',
    author_email='j.c.p.v.wanrooij@student.tue.nl',
    license='MIT',
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        'numpy'
    ],
)
