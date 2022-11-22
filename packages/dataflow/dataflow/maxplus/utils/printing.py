
from dataflow.maxplus.types import TMPMatrix, TMPVector, TMPVectorList, TTimeStamp
from dataflow.maxplus.algebra import MP_MINUSINFINITY


def mpElementToString(x: TTimeStamp, miStr: str = '-inf')->str:
    '''Return a 6-character wide string representation of the max-plus element x, using miStr, defaulting to '-inf' to represent minus infinity.'''
    # TODO: make the width and formatting more flexible. Not even sure it is 6, based on miStr
    if x is MP_MINUSINFINITY:
        return '  '+miStr
    return '{}'.format(x)

def mpVectorToString(v: TMPVector)->str:
    '''Return string representation of the vector v.'''
    return '[ {} ]'.format(' '.join([mpElementToString(x) for x in v]))

def printMPMatrix(M: TMPMatrix):
    '''Print matrix M to the console.'''
    print('[', end="")
    print('\n'.join([mpVectorToString(row) for row in M]), end="")
    print(']')

def printMPVectorList(vl: TMPVectorList):
    '''Print list of vectors to the console.'''
    print('[', end="")
    print('\n'.join(mpVectorToString(v) for v in vl), end="")
    print(']')

# pretty printing, consider the maximum prime factor and the number of prime factors (use their product?) of th common denominator to determine the 'complexity' of fractions, before switching to decimal.
