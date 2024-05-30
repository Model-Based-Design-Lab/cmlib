""" miscellaneous utility functions """

from string import digits
from fractions import Fraction

def only_digits(s: str)->str:
    """Filter only digits from string."""
    return ''.join(c for c in s if c in digits)

def onlyNonDigits(s):
    return ''.join(c for c in s if c not in digits)


def getIndex(name):
    dig = only_digits(name)
    if dig == '':
        return -1
    else:
        return int(dig)

def isNumerical(setOfNames):
    alphNames = set([onlyNonDigits(s) for s in setOfNames])
    return len(alphNames) <= 1

def sortNames(setOfNames):
    listOfNames = list(setOfNames)
    if isNumerical(setOfNames):
        listOfNames.sort(key=getIndex)
    else:
        listOfNames.sort()
    return listOfNames

def print4F(data):
    if type(data) == dict:
        for key in data:
            data[key] = print4F(data[key])
        return data

    if type(data) == list:
        string = "["
        for item in data:
            string += "{}, ".format(print4F(item))
        if len(string[:-2]) > 0:
            return string[:-2] + "]"
        else:
            return ""

    else:
        if not (type(data) == int or type(data) == float):
            try:
                xyz = "{}".format(data)
            except:
                print("***")
                print(type(data))
                print(data)
                return "XYZ"

            return "{}".format(data)
        else:
            return "{:.4f}".format(data)


