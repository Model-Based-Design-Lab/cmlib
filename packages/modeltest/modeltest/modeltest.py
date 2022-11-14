import os
import json
import numpy as np
import warnings
from modeltest.utils.utils import sortNames, print4F

class Model_pytest(object):

    def __init__(self, output_loc):
        self.output_loc = output_loc

        # Read / create output file
        self._create_output_file()
        self._read_output_file()

    def _change_type_of_data(self, data, sort, quotes):
        input_type = type(data)

        # First translate all numpy types to normal types
        if input_type.__module__ == np.__name__:
            if input_type == np.ndarray: # arrays/matrices
                data = data.tolist()
            else:
                data = data.item()    # Single data types (int/float...)
            input_type = type(data)

        if quotes and isinstance(data, str):
            return '\"' + data + '\"'

        if (not hasattr(data, "__iter__")) or isinstance(data, str):
            return data

        if input_type == dict:
            for key in data:
                data[key] = self._change_type_of_data(data[key], sort, quotes)
            return data

        # convert to list
        data = list(data)

        # Sorting algorithms
        if sort:
            try:
                data = sortNames(data)
            except:
                pass
            
            try:
                data = sorted(data)
            except:
                pass

        for n in range(len(data)):
            data[n] = self._change_type_of_data(data[n], sort, quotes)

        return data

    # translates any result to string format in order
    def _translate_to_str(self, data, sort, quotes):
        # Change type to either str,list,float or int
        data = self._change_type_of_data(data, sort, quotes)

        # Only at the very end round number to 4 decimal floats
        data = print4F(data) 

        return data

    def _create_output_file(self):
        if not os.path.exists(self.output_loc):
            with open(self.output_loc, 'w'): pass

    def _read_output_file(self):
        try:
            with open(self.output_loc, 'r') as outputFile:
                self.output = json.load(outputFile)
        except Exception:
            # Json file is empty
            self.output = {}

    def _single_cfunction_warning(self, func, name, sort, quotes):
        # Function to generate warning when expected result is not inside json file
        result = self._translate_to_str(func(), sort, quotes)

        if name in self.output:
            expected_result = self.output[name]
        else:
            self.output[name] = [result]
            expected_result = self.output[name]

        item_in_list = False
        for item in self.output[name]:
            if result == item:
                item_in_list = True
        
        if not item_in_list:
            warnings.warn("{}: {} not in {}".format(name, result, expected_result), Warning)
            self.output[name].append(result)

    def _single_cfunction_test(self, func, name, sort, quotes):
        # lambda function defined in higher class
        result = self._translate_to_str(func(), sort, quotes)
    
        if name in self.output:
            expected_result = self.output[name]
        else:
            self.output[name] = result
            expected_result = self.output[name]
        
        assert result == expected_result

    def incorrect_test(self, func, expected_result):
        # Argument always necessary for failure of function execution
        try:
            func()
            assert False
        except Exception as e:
            assert str(e) == expected_result
    
    def function_test(self, func, expected_result, deterministic = True, sort = False, quotes = False):
        if deterministic:
            self._single_cfunction_test(func, expected_result, sort, quotes)
        else:
            self._single_cfunction_warning(func, expected_result, sort, quotes)


    def write_output_file(self):
        with open(self.output_loc, 'w') as outputFile:
            json.dump(self.output, outputFile, indent = 4)
