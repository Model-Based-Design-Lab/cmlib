'''Model testing support.'''
import os
import json
import warnings
from modeltest.utils.utils import sort_names, print_4f

class ModelPytest:
    '''Model testing support.'''

    def __init__(self, output_loc):
        self.output_loc = output_loc

        # Read / create output file
        self._create_output_file()
        self._read_output_file()

    def _change_type_of_data(self, data, sort, quotes):
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
                data = sort_names(data)
            except TypeError:
                pass

            try:
                data = sorted(data)
            except TypeError:
                pass

        for n, d in enumerate(data):
            data[n] = self._change_type_of_data(d, sort, quotes)

        return data

    # translates any result to string format in order
    def _translate_to_str(self, data, sort, quotes):
        # Change type to either str,list,float or int
        data = self._change_type_of_data(data, sort, quotes)

        # Only at the very end round number to 4 decimal floats
        data = print_4f(data)

        return data

    def _create_output_file(self):
        if not os.path.exists(self.output_loc):
            with open(self.output_loc, 'w', encoding='utf-8'):
                pass

    def _read_output_file(self):
        try:
            with open(self.output_loc, 'r', encoding='utf-8') as output_file:
                self.output = json.load(output_file)
        except FileNotFoundError:
            # Json file is empty
            self.output = {}
        except json.JSONDecodeError:
            # Json file is empty
            self.output = {}

    def _single_c_function_warning(self, func, name, sort, quotes):
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
            warnings.warn(f"{name}: {result} not in {expected_result}", Warning)
            self.output[name].append(result)

    def _single_c_function_test(self, func, name, sort, quotes):
        # lambda function defined in higher class
        print(f"sort: {sort}")
        result = self._translate_to_str(func(), sort, quotes)
        if name in self.output:
            expected_result = self.output[name]
        else:
            self.output[name] = result
            expected_result = self.output[name]

        if not result == expected_result:
            print("Expected result: " + str(expected_result))
            print("Actual result: " + str(result))

        assert result == expected_result

    def incorrect_test(self, func, expected_result):
        '''Test behavior which is expected to fail.'''
        # Argument always necessary for failure of function execution
        try:
            func()
            assert False
        except Exception as e: # pylint: disable=broad-exception-caught
            print("Expected result: " + expected_result)
            print("Actual result: " + str(e))
            assert str(e) == expected_result

    def function_test(self, func, test_name, deterministic = True, sort = False, \
                      quotes = False):
        '''Test a function that is expected to succeed.'''
        if deterministic:
            self._single_c_function_test(func, test_name, sort, quotes)
        else:
            self._single_c_function_warning(func, test_name, sort, quotes)


    def write_output_file(self):
        '''Write self.output to file.'''
        with open(self.output_loc, 'w', encoding='utf-8') as output_file:
            json.dump(self.output, output_file, indent = 4)
