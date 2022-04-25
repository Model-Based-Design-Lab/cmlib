""" miscellaneous utility functions """

from string import digits

def onlyDigits(s):
    return ''.join(c for c in s if c in digits)

def onlyNonDigits(s):
    return ''.join(c for c in s if c not in digits)


def getIndex(name):
    dig = onlyDigits(name)
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
        if type(data) == bool or type(data) == str or data == None:
            return "{}".format(data)
        else:
            return "{:.4f}".format(data)


