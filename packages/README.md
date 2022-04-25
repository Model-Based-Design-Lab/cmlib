# Analysis Tool Packages

## Prerequisites

Make sure that you have python 3.x.x installed, including pip.

## Install

Any of the packages can be installed as follows.
**Make sure to install the python-graph package from the github location, as described below, not from the standard Python repository.**

In a console (shell, command prompt or powershell), in this folder, type:

``` shell
python -m pip install "git+https://github.com/Shoobx/python-graph#egg=pkg&subdirectory=core"
cd modeltest
python -m pip install .
cd ../markovchains
python -m pip install .
cd ../finitestateautomata
python -m pip install .
cd ../dataflow
python -m pip install .
```

Note that, depending on the python installation, you may need to have administrator / sudo rights to install the package.

## Running Tools

If the package is installed successfully, the analysis tools can be run as follows.

``` shell
markovchains
finitestateautomata
dataflow
```
