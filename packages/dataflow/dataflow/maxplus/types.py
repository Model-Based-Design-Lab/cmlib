'''Type definitions for max-plus algebra'''

from fractions import Fraction
from typing import List

TimeStamp = Fraction | None
TimeStampList = List[TimeStamp]
TimeStampFloat = float | None
TimeStampFloatList = List[TimeStampFloat]
MPVector = TimeStampList
MPVectorList = List[MPVector]
MPMatrix = MPVectorList
