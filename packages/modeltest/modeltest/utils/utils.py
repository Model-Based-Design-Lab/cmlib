""" miscellaneous utility functions """

from string import digits

def only_digits(s: str)->str:
    """Filter only digits from string."""
    return ''.join(c for c in s if c in digits)

def only_non_digits(s):
    """Filter only non-digits from string."""
    return ''.join(c for c in s if c not in digits)


def get_index(name):
    "Retrieve index number from name."
    dig = only_digits(name)
    if dig == '':
        return -1
    return int(dig)

def is_numerical(set_of_names):
    '''Test if all names have some digits.'''
    alpha_names = {only_non_digits(s) for s in set_of_names}
    return len(alpha_names) <= 1

def sort_names(set_of_names):
    '''Sort the set of names as a list.'''
    list_of_names = list(set_of_names)
    if is_numerical(set_of_names):
        list_of_names.sort(key=get_index)
    else:
        list_of_names.sort()
    return list_of_names

def print_4f(data):
    '''Print as .4f float.'''
    if isinstance(data, dict):
        for key in data:
            data[key] = print_4f(data[key])
        return data

    if isinstance(data, list):
        string = "["
        for item in data:
            string += f"{print_4f(item)}, "
        if len(string[:-2]) > 0:
            return string[:-2] + "]"
        return ""

    if isinstance(data, bool) or not isinstance(data,(int, float)):
        try:
            f_data = f"{data}"
        except ValueError:
            print("***")
            print(type(data))
            print(data)
            return "XYZ"

        return f_data
    return f"{data:.4f}"
