# Application test setup

The CMWB is supported with a functional testing environment to support continues development. The platform is called PyTest. A glue layer has been created to facilitate testing the three tools (FSA, Markovchains and Dataflow). This glue software is included in this directory.

## Prerequisites and installation

Make sure that you have python 3.x.x installed, including pip. Also, install the three tools as explained in [packages/README.md](../README.md). Install the following packages for python.

``` shell
python3.x -m pip install pytest
```

Then, navigate to the folder containing this README.md file and install the package called modeltest.

``` shell
python3.x -m pip install .
```

## Running the tests

Each tool can be tested separately by navigating to their corresponding directories: [packages/dataflow](../dataflow/), [packages/finitestateautomata](../finitestateautomata/) or [packages/markovchains](../markovchains/). From here, execute the following line in a terminal.

``` shell
python3.x -m pytest . -v -l
```

This will search through every python file with the name ```test_*``` or ```*_test``` and will execute any function with the name ```test_*``` or ```*_test```. The directory of every tool contains such a file, e.g. [packages/markovchains/markovchains/tests/test_markov.py](../markovchains/markovchains/tests/test_markov.py). This file contains all the functions that are tested for the Markovchains tool. The ```Markov_pytest``` class contains two functions, ```Correct_behavior_tests``` and ```Incorrect_behavior_tests```. The ```Correct_behavior_tests``` function checks if the functionality of the Markovchain tool remains correct. The ```Incorrect_behavior_tests``` tests if incorrect inputs remain incorrect. Tests can easily be added to either functions. The result of the ```Correct_behavior_tests``` can be found in the output directory, i.g. [packages/markovchains/markovchains/tests/output/](../markovchains/markovchains/tests/output). The name of each key in the json file corresponds to the name given in the ```Correct_behavior_tests``` function.

## Adding function to tests

The tests in the ```Correct_behavior_tests``` function require a lambda function and a name. The lambda function contains the tool function you want to test. Take the markovchain tool as an example. The name is used to store the answer in [packages/markovchains/markovchains/tests/output/]. The lambda function includes the markovchain function that will be tested, with its corresponding variables. Take the following example for testing the functionality of the ```executeSteps()``` function from the markovchain tool. With the following test, the ```executeSteps()``` function is tested for ```15``` steps. The result is compared with the result saved in the [packages/markovchains/markovchains/tests/output/](../markovchains/markovchains/tests/output) file.

``` shell
self.function_test(lambda: self.model.executeSteps(15), "executeSteps_15")
```

The ```Incorrect_behavior_tests()``` function requires a lambda function and an expected result as inputs. This is done to ensure that the engineer adding incorrect tests do not have to check the output file for the correctness of the fault message.

## Updating Test Reference Output

If the reference output JSON files of the test need to be updated, this can be achieved by deleting the old reference files and running the test. It will regenerate the reference output.
